import argparse
import os
import numpy as np
import splitfolders
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

# splitfolder를 활용하여 flower 폴더 내 이미지 데이터 분류 및 train, val, test 정렬
def split_dataset(data_path):
    splitfolders.ratio(f'{data_path}/flowers', output=f'{data_path}/dataset', seed=42, ratio=(0.7, 0.15, 0.15))
    # print data amount
    data = ['train', 'val', 'test']
    label = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    for i in data:
        for j in label:
            count = len(os.listdir(f'{data_path}/dataset/{i}/{j}'))
            print(f'Crack | {i} | {j} : {count}')

def get_mean_std(data_path, img_size):
    class Transforms:
        def __init__(self, transforms: A.Compose):
            self.transforms = transforms
        def __call__(self, img, *args, **kwargs):
            return self.transforms(image=np.array(img))['image']
    dataset = datasets.ImageFolder(f'{data_path}/flowers', transform=Transforms(transforms=A.Compose([A.Resize(img_size , img_size), ToTensorV2()])))
    loader = DataLoader(dataset, batch_size=10, num_workers=0, shuffle=False)
    mean = 0.0
    for images, _ in loader:
        images = images / 255
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(loader.dataset)
    var = 0.0
    for images, _ in loader:
        images = images / 255
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(loader.dataset)*img_size*img_size))
    mean = list(map(lambda x: str(x) + '\n', mean.tolist()))
    std = list(map(lambda x: str(x) + '\n', std.tolist()))
    with open(f'{data_path}/mean-std.txt', 'w') as f:
        f.writelines(mean)
        f.writelines(std)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--img-size', type=int, help='resize img size')
    opt = parser.parse_args()
    split_dataset(opt.data_path)
    get_mean_std(opt.data_path, opt.img_size)