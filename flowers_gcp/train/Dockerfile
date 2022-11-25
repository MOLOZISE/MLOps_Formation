FROM pytorch/pytorch:latest
RUN pip install -U numpy albumentations tqdm timm mlflow boto3
RUN mkdir -p /app
ARG DISABLE_CACHE
ADD . /app/
WORKDIR /app
ENTRYPOINT ["python", "train.py" ]