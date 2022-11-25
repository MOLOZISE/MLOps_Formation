import kfp
from kfp import dsl
from kfp import onprem
def preprocess_op(pvc_name, volume_name, volume_mount_path):
    return dsl.ContainerOp(
        name='Preprocess Data',
        image='molozise/kfp-flowers-preprocess:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))
def hyp_op(pvc_name, volume_name, volume_mount_path, device):
    return dsl.ContainerOp(
        name='Hyperparameter Tuning',
        image='molozise/kfp-flowers-hyperparameter:latest',
        arguments=['--data-path', volume_mount_path,
                   '--device', device],
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path))
def train_op(pvc_name, volume_name, volume_mount_path, device):
    return dsl.ContainerOp(
        name='Train Model',
        image='molozise/kfp-flowers-train:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-name', 'surface-ConvNeXt-T',
                   '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)
def test_op(pvc_name, volume_name, volume_mount_path, model_path, device):
    return dsl.ContainerOp(
        name='Test Model',
        image='molozise/kfp-flowers-test:latest',
        arguments=['--data-path', volume_mount_path,
                   '--img-size', 224,
                   '--model-path', model_path,
                   '--device', device]
    ).apply(onprem.mount_pvc(pvc_name, volume_name=volume_name, volume_mount_path=volume_mount_path)).set_gpu_limit(4)
@dsl.pipeline(
    name='Flowers Pipeline',
    description=''
)
def surface_pipeline(mode_hyp_train_test: str,
                     preprocess_yes_no: str,
                     model_path: str,
                     device: str):
    pvc_name = "workspace-flowers"
    volume_name = 'pipeline'
    volume_mount_path = '/home/jovyan'
    with dsl.Condition(preprocess_yes_no == 'yes'):
        _preprocess_op = preprocess_op(pvc_name, volume_name, volume_mount_path)
    with dsl.Condition(mode_hyp_train_test == 'hyp'):
        _hyp_op = hyp_op(pvc_name, volume_name, volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'train'):
        _train_op = train_op(pvc_name, volume_name, volume_mount_path, device).after(_preprocess_op)
    with dsl.Condition(mode_hyp_train_test == 'test'):
        _train_op = test_op(pvc_name, volume_name, volume_mount_path, model_path, device).after(_preprocess_op)
if __name__ == '__main__':
    kfp.compiler.Compiler().compile(surface_pipeline, './flowers_pipeline.yaml')