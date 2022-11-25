# Katib 대신 weights and biases(wandb)를 활용

import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR
#
from util import seed_everything, MetricMonitor, build_dataset, build_optimizer
import wandb

class ConvNeXt(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ConvNeXt, self).__init__()
        self.model = timm.create_model('convnext_tiny', pretrained=pretrained)  # timm 라이브러리에서 pretrained model 가져옴
        self.model.head.fc = nn.Linear(self.model.head.fc.in_features, num_classes, bias=True)
    def forward(self, x):
        return self.model(x)

def train_epoch(train_loader, epoch, model, optimizer, criterion, device):
    metric_monitor = MetricMonitor()
    model.train()
    stream = tqdm(train_loader)
    for i, (images, targets) in enumerate(stream, start=1):
        images, targets = images.float().to(device), targets.to(device)
        output = model(images)
        loss = criterion(output, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        predicted = torch.argmax(output, dim=1)
        accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
        metric_monitor.update('Loss', loss.item())
        metric_monitor.update('accuracy', accuracy)
        stream.set_description(
            f"Epoch: {epoch}. Train. {metric_monitor}"
        )
        wandb.log({"Train Epoch": epoch, "Train loss": loss.item(), "Train accuracy": accuracy})
def val_epoch(val_loader, epoch, model, criterion, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(val_loader)
    val_loss = 0
    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images, targets = images.float().to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            val_loss += loss
            predicted = torch.argmax(output, dim=1)
            accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accuracy)
            stream.set_description(
                f"Epoch: {epoch}. Validation. {metric_monitor}"
            )
            wandb.log({"Validation Epoch":epoch, "Validation loss": loss.item(), "Validation accuracy": accuracy})
        wandb.log({"VAL EPOCH LOSS": val_loss / len(val_loader.dataset)})
    return accuracy
def main(hyperparameters=None):
    wandb.init(project='flowers-classification', config=hyperparameters)
    config = wandb.config
    epochs = 1
    # read mean std values
    with open(f'{opt.data_path}/mean-std.txt', 'r') as f:
        cc = f.readlines()
        mean_std = list(map(lambda x: x.strip('\n'), cc))
    model = ConvNeXt(num_classes=5, pretrained=True)
    model.to(device)
    train_loader, val_loader, _ = build_dataset(opt.data_path, config.img_size, config.batch_size, mean_std)
    optimizer = build_optimizer(model, config.optimizer, config.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=10,
                                  eta_min=1e-6,
                                  last_epoch=-1)
    for epoch in range(1, epochs + 1):
        train_epoch(train_loader, epoch, model, optimizer, criterion, device)
        val_epoch(val_loader, epoch, model, criterion, device)
        scheduler.step()
def configure():
    sweep_config = \
    {'method': 'random',
     'metric': {'goal': 'minimize', 'name': 'VAL EPOCH LOSS'},
     'parameters': {'batch_size': {'values': [32, 64, 128]},
                    'epochs': {'value': 1},
                    'img_size': {'values': [112, 224]},
                    'lr': {'distribution': 'uniform',
                                      'max': 0.1,
                                      'min': 0.001},
                    'optimizer': {'values': ['adam', 'sgd']}}}
    return sweep_config
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--device', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    if opt.device == 'cpu':
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE is {device}")
    seed_everything()
    wandb.login(key='19b5d5f540b701ccc26599e2e65cd2dbc1ab11d4')
    hyperparameters = configure()
    sweep_id = wandb.sweep(hyperparameters, project='flowers-classification')
    wandb.agent(sweep_id, main, count=10)  # count: 실험 횟수