from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
import torch
import sys
sys.path.append("/root/shared-nvme/ControlAR_subject")
import os
import time
import argparse
from torch.utils.data import DataLoader
from torchvision import transforms
from glob import glob
from utils.logger import create_logger
from dataset.augmentation import center_crop_arr
from autoregressive.models.gpt_t2i import GPT_models
from tokenizer.tokenizer_image.vq_model import VQ_models
from accelerate.utils import set_seed
from dataset.t2i_subject import build_t2i_subject_code
import torch._dynamo
import cv2
import inspect

torch._dynamo.config.suppress_errors = True

def creat_optimizer(model, weight_decay, learning_rate, betas, logger):
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    logger.info(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    logger.info(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    # fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    # extra_args = dict(fused=True) if fused_available else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    logger.info(f"using fused AdamW")
    return optimizer

def match_size(input_tensor):
    output_tensor = input_tensor.mean(dim=1, keepdim=True)

    return output_tensor

def main(args):
    # Set device and seed
    device = torch.device("cuda")
    set_seed(args.global_seed)

    # Setup an experiment folder
    os.makedirs(args.results_dir, exist_ok=True)
    experiment_index = len(glob(f"{args.results_dir}/*"))
    model_string_name = args.gpt_model.replace("/", "-")
    experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
    checkpoint_dir = f"{experiment_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger = create_logger(experiment_dir)
    logger.info(f"Experiment directory created at {experiment_dir}")

    # Setup model
    latent_size = args.image_size // args.downsample_size
    model = GPT_models[args.gpt_model](
        vocab_size=args.vocab_size,
        block_size=latent_size ** 2,
        num_classes=args.num_classes,
        cls_token_num=args.cls_token_num,
        model_type=args.gpt_type,
        resid_dropout_p=args.dropout_p,
        ffn_dropout_p=args.dropout_p,
        token_dropout_p=args.token_dropout_p,
        condition_type=args.condition_type,
    ).to(device)
    logger.info(f"GPT Parameters: {sum(p.numel() for p in model.parameters()):,}")
    # Setup optimizer
    optimizer = creat_optimizer(model, args.weight_decay, args.lr, (args.beta1, args.beta2), logger)

    # Prepare dataset and loader
    train_dataset = build_t2i_subject_code(args)
    loader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        batch_size=args.global_batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(train_dataset):,} images")

    # Load checkpoint if provided
    if args.gpt_ckpt:
        checkpoint = torch.load(args.gpt_ckpt, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
        train_steps = 0  # checkpoint.get("steps", 0)
        start_epoch = 0  # checkpoint.get("epoch", 0)
        del checkpoint
        logger.info(f"Resume training from checkpoint: {args.gpt_ckpt}")
    else:
        train_steps = 0
        start_epoch = 0

    if not args.no_compile:
        logger.info("Compiling the model... (may take several minutes)")
        model = torch.compile(model)  # requires PyTorch 2.0

    model.train()  # Enable dropout for training

    # Training loop
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        print(len(train_dataset.images_list))
        for batch in loader:
            x = batch['code']
            print(x.size())
            caption_emb = batch['caption_emb']
            print(caption_emb.size())
            subject_img = batch['subject']
            print(subject_img.size())
            attn_mask = batch['attn_mask']
            valid = batch['valid']

            x = x.to(device, non_blocking=True)
            caption_emb = caption_emb.to(device, non_blocking=True)
            subject_img = subject_img.to(device, non_blocking=True)
            attn_mask = attn_mask.to(device)
            valid = valid.to(device)
            with torch.no_grad():
                subject_img = match_size(subject_img.float()).repeat(1, 3, 1, 1)
                subject_img = 2 * (subject_img - 0.5)

            z_indices = x.reshape(x.shape[0], -1)
            c_indices = caption_emb.reshape(caption_emb.shape[0], caption_emb.shape[-2], caption_emb.shape[-1])
            attn_mask = attn_mask.reshape(attn_mask.shape[0], 1, attn_mask.shape[-2], attn_mask.shape[-1])

            _, loss = model(cond_idx=c_indices, idx=z_indices[:, :-1], targets=z_indices, mask=attn_mask[:, :, :-1, :-1], valid=valid, condition=subject_img)
            
            # Backward pass
            loss.backward()
            if args.max_grad_norm != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Log loss
            train_steps += 1
            # if train_steps % args.log_every == 0:
            logger.info(f"(step={train_steps:07d}) Loss: {loss.item():.4f}")

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                model_weight = model.state_dict()
                checkpoint = {
                    "model": model_weight,
                    "steps": train_steps,
                    "args": args
                }
                checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()
    logger.info("Training complete!")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lst-dir", type=str, default="/root/shared-nvme/data/Subjects200K/Subjects200K.json")
    parser.add_argument("--code-path", type=str, default="/root/shared-nvme/ControlAR_subject/code")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--caption-dir", type=str, default="/root/shared-nvme/ControlAR_subject/caption_embds")
    parser.add_argument("--t5-feat-path", type=str, required=False)
    parser.add_argument("--short-t5-feat-path", type=str, default=None, help="short caption of t5_feat_path")
    parser.add_argument("--no-local-save", action='store_true', help='no save checkpoints to local path for limited disk volume')
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default=None, help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--gpt-model", type=str, choices=list(GPT_models.keys()), default="GPT-XL")
    parser.add_argument("--gpt-ckpt", type=str, default=None, help="ckpt path for resume training")
    parser.add_argument("--gpt-type", type=str, choices=['c2i', 't2i'], default="t2i")
    parser.add_argument("--vocab-size", type=int, default=16384, help="vocabulary size of visual tokenizer")
    parser.add_argument("--cls-token-num", type=int, default=120, help="max token number of condition input")
    parser.add_argument("--dropout-p", type=float, default=0.1, help="dropout_p of resid_dropout_p and ffn_dropout_p")
    parser.add_argument("--token-dropout-p", type=float, default=0.1, help="dropout_p of token_dropout_p")
    parser.add_argument("--drop-path", type=float, default=0.0, help="drop_path_rate of attention and ffn")
    parser.add_argument("--no-compile", action='store_true')
    parser.add_argument("--results-dir", type=str, default="/root/shared-nvme/ControlAR_subject/results")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 512], default=512)
    parser.add_argument("--downsample-size", type=int, choices=[8, 16], default=16)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2, help="Weight decay to use.")
    parser.add_argument("--beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.95, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--max-grad-norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--global-batch-size", type=int, default=1)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--mixed-precision", type=str, default='fp16', choices=["none", "fp16", "bf16"]) 
    
    parser.add_argument("--condition-type", type=str, choices=['canny', 'hed', 'lineart', 'depth'], default="lineart")
    parser.add_argument("--t5-feature-max-len", type=int, default=120)
    parser.add_argument("--t5-feature-dim", type=int, default=2048)
    parser.add_argument("--keep_in_memory",type=bool,default=False)
    parser.add_argument("--wrong_ids_file",type=str,default=None)
    parser.add_argument("--logging_dir",type=str,default="logs")
    args = parser.parse_args()
    main(args) 