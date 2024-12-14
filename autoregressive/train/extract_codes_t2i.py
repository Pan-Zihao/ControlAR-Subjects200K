import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import numpy as np
from PIL import Image
import glob
import argparse
import os
import json
import sys
sys.path.append("./")
from utils.distributed import init_distributed_mode
from dataset.augmentation import center_crop_arr
from tokenizer.tokenizer_image.vq_model import VQ_models

class CustomDataset(Dataset):
    def __init__(self, lst_dir, transform, num, code_dir):
        subject_list = []
        images_list = []
        self.image2subject = {}
        with open(lst_dir,'r') as file:
            data = json.load(file)
        for subject in data.keys():
            subject_list.append(subject)
        self.subject_list = subject_list[:num]
        for subject in self.subject_list:
            for image_path in data[subject].keys():
                images_list.append(image_path)
                self.image2subject[image_path] = subject
        codes_list = list(range(len(images_list)))  
        self.map_image2code = dict(zip(images_list, codes_list))
        self.images_list = images_list
        self.transform = transform
        self.code_dir = code_dir

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_path = self.images_list[index]
        code_path = os.path.join(self.code_dir, f'{self.map_image2code[img_path]}.npy')
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, code_path

def main(args):

    # assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    '''
    os.environ['MASTER_ADDR'] = '0.0.0.0'
    os.environ['MASTER_PORT'] = '12345'
    os.environ['WORLD_SIZE'] = '8'  # Set to the total number of GPUs
    os.environ['RANK'] = '0'  # Default to rank 0 for local training setup

    # Setup DDP:
    dist.init_process_group(backend='nccl', init_method='env://')
    init_distributed_mode(args)
    rank = dist.get_rank()
    device = torch.device(f'cuda:{rank % torch.cuda.device_count()}')
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    '''
    device = torch.device("cuda")
    # create and load model
    vq_model = VQ_models[args.vq_model](
        codebook_size=args.codebook_size,
        codebook_embed_dim=args.codebook_embed_dim)
    vq_model.to(device)
    vq_model.eval()
    checkpoint = torch.load(args.vq_ckpt, map_location="cpu")
    vq_model.load_state_dict(checkpoint["model"])
    del checkpoint

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.json_path, transform=transform, num=args.num, code_dir=args.code_path)
    '''
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=False,
        seed=args.global_seed
    )
    '''
    loader = DataLoader(
        dataset,
        batch_size=1,  # important!
        shuffle=False,
        # sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    # Processing images
    for img, code_path in loader:
        img = img.to(device)
        with torch.no_grad():
            _, _, [_, _, indices] = vq_model.encode(img)
        codes = indices.reshape(img.shape[0], -1)
        x = codes.detach().cpu().numpy()    # (1, args.image_size//16 * args.image_size//16)
        np.save(code_path[0], x)

        print(f"Processed and saved: {code_path[0]}")

    # dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="/root/shared-nvme/data/Subjects200K/Subjects200K.json")
    parser.add_argument("--code-path", type=str, default="/root/shared-nvme/ControlAR_subject/code")
    parser.add_argument("--num", default=5, type=int)
    parser.add_argument("--vq-model", type=str, choices=list(VQ_models.keys()), default="VQ-16")
    parser.add_argument("--vq-ckpt", type=str, default="/root/shared-nvme/ControlAR_subject/checkpoints/vq/vq_ds16_t2i.pt", help="ckpt path for vq model")
    parser.add_argument("--codebook-size", type=int, default=16384, help="codebook size for vector quantization")
    parser.add_argument("--codebook-embed-dim", type=int, default=8, help="codebook dimension for vector quantization")
    parser.add_argument("--image-size", type=int, choices=[256, 384, 448, 512], default=512)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    args = parser.parse_args()
    main(args)
    '''
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = CustomDataset(args.json_path, transform=transform, num=args.num, code_dir=args.code_path)
    print(len(dataset.images_list))
    print(dataset.map_image2code)
    '''
    
