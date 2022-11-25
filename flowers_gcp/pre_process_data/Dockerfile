FROM pytorch/pytorch:latest
RUN pip install -U numpy split-folders albumentations
RUN mkdir -p /app
ARG DISABLE_CACHE
ADD pre_process_data.py /app/
WORKDIR /app
ENTRYPOINT ["python", "pre_process_data.py" ]