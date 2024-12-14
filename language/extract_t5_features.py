import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import os
import json

from t5 import T5Embedder

class CustomDataset(Dataset):
    def __init__(self, lst_dir, num, code_dir):
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
        self.code_dir = code_dir

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        img_path = self.images_list[index]
        code_path = os.path.join(self.code_dir, f'{self.map_image2code[img_path]}.npy')
        return img_path, code_path
    
def main(args):
    device = torch.device("cuda")

    # Setup data:
    print(f"Dataset is preparing...")
    dataset = CustomDataset(args.json_path, args.num, args.code_dir)
    loader = DataLoader(
        dataset,
        batch_size=1, # important!
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )
    print(f"Dataset contains {len(dataset):,} images")

    precision = {'none': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[args.precision]
    assert os.path.exists(args.t5_model_path)
    t5_xxl = T5Embedder(
        device=device, 
        local_cache=True, 
        cache_dir=args.t5_model_path, 
        dir_or_name=args.t5_model_type,
        torch_dtype=precision
    )
    with open(args.json_path,'r') as file:
            data = json.load(file)
    map = dataset.image2subject
    for img_path, code_path in loader:
        subject = map[img_path[0]]
        print(subject)
        print(img_path[0])
        caption = data[subject][img_path[0]]
        caption_embs, emb_masks = t5_xxl.get_text_embeddings(caption)
        valid_caption_embs = caption_embs[:, :emb_masks.sum()]
        x = valid_caption_embs.to(torch.float32).detach().cpu().numpy()
        print("输出的特征维度是", x.shape)
        np.save(code_path[0], x)
        print(code_path[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-path", type=str, default="/root/shared-nvme/data/Subjects200K/Subjects200K.json")
    parser.add_argument("--t5-path", type=str, default="/root/shared-nvme/ControlAR_subject/checkpoints/t5-ckpt")
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--t5-model-path", type=str, default='/root/shared-nvme/ControlAR_subject/checkpoints/t5-ckpt')
    parser.add_argument("--t5-model-type", type=str, default='flan-t5-xl')
    parser.add_argument("--precision", type=str, default='bf16', choices=["none", "fp16", "bf16"])
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--code-dir", type=str, default="/root/shared-nvme/ControlAR_subject/caption_embds")
    args = parser.parse_args()
    main(args)