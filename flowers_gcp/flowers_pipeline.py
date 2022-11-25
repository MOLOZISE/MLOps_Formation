#!pip3 install -U kfp
import kfp
import kfp.components as comp
from kfp import dsl
from kfp import compiler
from kfp.components import func_to_container_op
import time
import datetime
# Function for determine deployment
import kfp
from kfp import dsl

def preprocess_op(volume_mount_path):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='molozise/kfp-flowers-gcp-preprocess:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224],
    )
def hyp_op(volume_mount_path, device):
    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='molozise/kfp-flowers-gcp-hyperparameter:latest',
        arguments=['--data-path', volume_mount_path,
                   '--device', device],
    )
def train_op(volume_mount_path, device):
    return dsl.ContainerOp(
        name='Train Model',
        image='molozise/kfp-flowers-gcp-train:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-name', 'flowers-ConvNeXt-T',
                   '--device', device]
    )
def test_op(volume_mount_path, model_path, device):
    return dsl.ContainerOp(
        name='Test Model',
        image='molozise/kfp-flowers-gcp-test:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-path', model_path,
                   '--device', device]
    )
@dsl.pipeline(
    name='Flowers Pipeline',
    description=''
)
def flowers_pipeline(mode_hyp_train_test: str,
                     preprocess_yes_no: str,
                     model_path: str,
                     device: str):
    PIPELINE_HOST = "15bf934d55d3d679-dot-us-central1.pipelines.googleusercontent.com"  # Kubeflow Pipeline URL
    WORK_BUCKET = "gs://vertex-ai-example-368505-kubeflowpipelines-default"  # Cloud Storage Bucket
    EXPERIMENT_NAME = "Flowers Classification Experiment"  # Experiment Name
    volume_mount_path = WORK_BUCKET
    with dsl.Condition(preprocess_yes_no == 'yes'):
        _preprocess_op = preprocess_op(volume_mount_path)
    with dsl.Condition(mode_hyp_train_test == 'hyp'):
        _hyp_op = hyp_op(volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'train'):
        _train_op = train_op(volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'test'):
        _train_op = test_op(volume_mount_path, model_path, device).after(_preprocess_op)
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './flowers_pipeline.yaml')