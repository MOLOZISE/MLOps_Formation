import os
from collections import defaultdict
import numpy as np
import torch
import torch.optim as optim
from torchvision import datasets
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
class Transforms:
    def __init__(self, img_size, mean_std, data='train'):
        self.transforms = self.image_transform(img_size, mean_std, data)
    def image_transform(self, img_size, mean_std, data):
        if data == 'train':
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.HorizontalFlip(p=0.5),
                    A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2(),
                ]
            )
        elif data == 'val':
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2(),
                ]
            )
        # (TTA)Test Time Augmentation
        elif data == 'test':
            return A.Compose(
                [
                    A.Resize(img_size, img_size),
                    A.Normalize((mean_std[0], mean_std[1], mean_std[2]), (mean_std[3], mean_std[4], mean_std[5])),
                    ToTensorV2(),
                ]
            )
    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))['image']
class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()
    def reset(self):
        self.metrics = defaultdict(lambda: {"val": 0, "count": 0, "avg": 0})
    def update(self, metric_name, val):
        metric = self.metrics[metric_name]
        metric["val"] += val
        metric["count"] += 1
        metric["avg"] = metric["val"] / metric["count"]
    def __str__(self):
        return " | ".join(
            [
                "{metric_name}: {avg:.{float_precision}f}".format(
                    metric_name=metric_name, avg=metric["avg"],
                    float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
def build_dataset(data_path, img_size, batch_size, mean_std):
    train_dataset = datasets.ImageFolder(f'{data_path}/dataset/train', transform=Transforms(img_size, mean_std, 'train'))
    val_dataset = datasets.ImageFolder(f'{data_path}/dataset/val', transform=Transforms(img_size, mean_std, 'val'))
    test_dataset = datasets.ImageFolder(f'{data_path}/dataset/test', transform=Transforms(img_size, mean_std, 'test'))
    train_loader = DataLoader(train_dataset, batch_size,
                              num_workers=0, pin_memory=True, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size,
                            num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size,
                             num_workers=0, pin_memory=True, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader
def build_optimizer(model, optimizer, learning_rate):
    if optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
    return optimizer