from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import Dataset
import json


class Subjects200K(Dataset):
    def __init__(self, dataset_dir, json_path):
        self.dataset_dir = dataset_dir
        self.Subjects = {}
        self.subjects = []
        self.index = 0
        self.json = json_path
    
    def get_subjects(self):
        dataset = load_dataset(self.dataset_dir)
        data = dataset['train']
        for row in tqdm(data):
            subject = row['description']['item']
            if subject not in self.subjects:
                self.subjects.append(subject)
            images_set = self.get_one_subject(row)
            if subject not in self.Subjects:
                self.Subjects[subject] = images_set
            else:
                self.Subjects[subject].update(images_set)
        print(len(self.subjects))
    
    def get_images(self,image):
        new_width = image.width // 2
        left_image = image.crop((0, 0, new_width, image.height))
        right_image = image.crop((new_width, 0, image.width, image.height))
        left_image_path = f"./Subjects200K_images/{self.index}.jpg"
        self.index = self.index + 1
        right_image_path = f"./Subjects200K_images/{self.index}.jpg"
        self.index = self.index + 1
        left_image.save(left_image_path)
        right_image.save(right_image_path)
        return left_image_path, right_image_path
    
    def get_one_subject(self,row):
        images_set = {}
        path_1, path_2 = self.get_images(row['image'])
        caption_1 = row['description']['description_0']
        caption_2 = row['description']['description_1']
        images_set[path_1] = caption_1
        images_set[path_2] = caption_2
        return images_set
    
    def __len__(self):
        return len(self.Subjects)
    
    # def __getitem__(self, idx):
    
if __name__ == "main":
    dataset_dir = "./Subjects200K"
    dataset = Subjects200K(dataset_dir, "./Subjects200K.json")
    # 初始化数据集，只需要一次即可
    dataset.get_subjects()
    with open("Subjects200K.json", 'w') as file:
    # 将字典转换为 JSON 格式并写入文件
        json.dump(dataset.Subjects, file, indent=4)


    

    
    