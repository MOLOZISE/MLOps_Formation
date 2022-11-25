import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
import mlflow
from util import seed_everything, MetricMonitor, build_dataset
def test(test_loader, model, criterion, device):
    metric_monitor = MetricMonitor()
    model.eval()
    stream = tqdm(test_loader)
    with torch.no_grad():
        for i, (images, targets) in enumerate(stream, start=1):
            images, targets = images.float().to(device), targets.to(device)
            output = model(images)
            loss = criterion(output, targets)
            predicted = torch.argmax(output, dim=1)
            accuracy = round((targets == predicted).sum().item() / targets.shape[0] * 100, 2)
            metric_monitor.update('Loss', loss.item())
            metric_monitor.update('accuracy', accuracy)
            stream.set_description(
                f"Test. {metric_monitor}"
            )
def main(opt, device):
    mlflow.set_tracking_uri("http://mlflow-server-service.mlflow-system.svc:5000")
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
    batch_size = 64
    with open(f'{opt.data_path}/mean-std.txt', 'r') as f:
        cc = f.readlines()
        mean_std = list(map(lambda x: x.strip('\n'), cc))
    model = mlflow.pytorch.load_model(opt.model_path)
    model.to(device)
    _, _, test_loader = build_dataset(opt.data_path, opt.img_size, batch_size, mean_std)
    criterion = nn.CrossEntropyLoss()
    test(test_loader, model, criterion, device)  # 마지막 iteration의 값들
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, help='dataset root path')
    parser.add_argument('--img-size', type=int, help='resize img size')
    parser.add_argument('--model-path', type=str, help='model path in mlflow, i,e. s3://~')
    parser.add_argument('--device', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    if opt.device == 'cpu':
        device = 'cpu'
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.device
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE is {device}")
    seed_everything()
    main(opt, device)