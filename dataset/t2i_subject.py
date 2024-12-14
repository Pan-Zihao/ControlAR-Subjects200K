from PIL import PngImagePlugin
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte
import torch
import random
import json
import argparse
from PIL import Image
from torchvision.transforms.functional import normalize
import sys
sys.path.append("/root/shared-nvme/ControlAR_subject")
import numpy as np
import os
from torch.utils.data import Dataset
import random
import argparse
from torchvision import transforms

transform = transforms.Compose([
    transforms.CenterCrop(512),  # 裁剪为 512x512 大小
])


class T2ICustomCode(Dataset):
    def __init__(self, args):
        # VQ编码后的image code路径
        self.code_dir = args.code_path
        # 图像大小
        self.image_size = args.image_size
        # 压缩过的code大小
        latent_size = args.image_size // args.downsample_size
        # 展平后的序列长度
        self.code_len = latent_size ** 2
        self.t5_feature_max_len = 120
        self.t5_feature_dim = 2048
        self.max_seq_length = self.t5_feature_max_len + self.code_len
        
        with open(args.lst_dir, 'r') as file:
            data = json.load(file)
        self.data = data
        subject_list = []
        images_list = []
        self.image2subject = {}
        for subject in data.keys():
            subject_list.append(subject)
        self.subject_list = subject_list[:args.num]
        for subject in self.subject_list:
            for image_path in data[subject].keys():
                images_list.append(image_path)
                self.image2subject[image_path] = subject
        codes_list = list(range(len(images_list)))  
        self.map_image2code = dict(zip(images_list, codes_list))
        self.images_list = images_list
        self.caption_dir = args.caption_dir
            
    
    def __len__(self):
        return len(self.images_list)
    
    def dummy_data(self):
        img = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool)).unsqueeze(0)
        valid = 0
        return img, t5_feat_padding, attn_mask, valid
    
    def collate_fn(self, examples):
        code = torch.stack([example["code"] for example in examples])
        subject =  torch.stack([example["subject"] for example in examples])
        caption_emb =  torch.stack([example["caption_emb"] for example in examples])
        attn_mask = torch.stack([example["attn_mask"] for example in examples])
        valid = torch.stack([example["valid"] for example in examples])
        output = {}
        output['code'] = code
        output['subject'] = subject
        output['caption_emb'] = caption_emb
        output['attn_mask'] = attn_mask
        output['valid'] = valid
        return output
    
    def __getitem__(self, index):
        print(index, "\n")
        img_path = self.images_list[index]
        code_path = os.path.join(self.code_dir, f'{self.map_image2code[img_path]}.npy')
        print(len(self.images_list))
        sub = self.image2subject[img_path]
        images_set = [item for item in self.data[sub].keys() if item != img_path]
        subject_path = random.choice(images_set)
        caption_path = os.path.join(self.caption_dir, f'{self.map_image2code[img_path]}.npy')
        code = np.load(code_path)
        subject = np.array(transform(Image.open(subject_path)))
        
        t5_feat_padding = torch.zeros((1, self.t5_feature_max_len, self.t5_feature_dim))
        caption = np.load(caption_path)
        t5_feat = torch.from_numpy(caption)
        t5_feat_len = t5_feat.shape[1] 
        feat_len = min(self.t5_feature_max_len, t5_feat_len)
        t5_feat_padding[:, -feat_len:] = t5_feat[:, :feat_len]
        emb_mask = torch.zeros((self.t5_feature_max_len,))
        emb_mask[-feat_len:] = 1
        attn_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length))
        T = self.t5_feature_max_len
        attn_mask[:, :T] = attn_mask[:, :T] * emb_mask.unsqueeze(0)
        eye_matrix = torch.eye(self.max_seq_length, self.max_seq_length)
        attn_mask = attn_mask * (1 - eye_matrix) + eye_matrix
        attn_mask = attn_mask.unsqueeze(0).to(torch.bool)
        valid = 1
        
        output = {}
        output['code'] = torch.from_numpy(code)
        output['subject'] = torch.from_numpy(subject.transpose(2,0,1))
        output['caption_emb'] = t5_feat_padding
        output['attn_mask'] = attn_mask
        output['valid'] = torch.tensor(valid)
        return output
    
def build_t2i_subject_code(args):
    dataset = T2ICustomCode(args)
    return dataset
'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lst-dir", type=str, default="/root/shared-nvme/data/Subjects200K/Subjects200K.json")
    parser.add_argument("--code-path", type=str, default="/root/shared-nvme/ControlAR_subject/code")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--downsample-size", type=int, default=16)
    parser.add_argument("--num", type=int, default=5)
    parser.add_argument("--caption-dir", type=str, default="/root/shared-nvme/ControlAR_subject/caption_embds")
    
    args = parser.parse_args()
    dataset = T2ICustomCode(args)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        batch_size=1,
        num_workers=1,
    )
    output = next(iter(dataloader))
    print("VQ编码后的图像:", output['code'], "\n形状是", output['code'].size())
    print("随机挑选的subject图像", output['subject'], "\n形状是", output['subject'].size())
    print("图像描述编码",output['caption_emb'],"\n形状是",output['caption_emb'].size())
    print(output['attn_mask'],"\n形状是",output['attn_mask'].size())
    print(output['valid']) 
'''

        
    

        
        
        
        
        
         
        

