
# 1. kubeflow pipeline 형식을 따름
# 2. mlflow에서 모델이 저장되도록 함
# 3. how : inference에서 모델 로딩 -> 로딩 부분만 추출하여 로딩 후 바로 저장하는 방식
# 4. 사전작업 : minio mlflow bucket 내 모델 업로딩 -> 모델 로딩 ->
# mlflow 형식으로 모델 저장(텐서플로 케라스, 파이토치 형식을 확인) -> Seldon Core로 배포


# SVC 모델 Train
# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.svm import SVC
#
# iris = load_iris()
#
# data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
# target = pd.DataFrame(iris["target"], columns=["target"])
#
# clf = SVC(kernel="rbf")
# clf.fit(data, target)


# MLFlow Infos

# from mlflow.models.signature import infer_signature
# from mlflow.utils.environment import _mlflow_conda_env
#
# input_example = data.sample(1)
# signature = infer_signature(data, clf.predict(data))
# conda_env = _mlflow_conda_env(additional_pip_deps=["dill", "pandas", "scikit-learn"])

# Save MLFlow Infos

# from mlflow.sklearn import save_model
#
# save_model(
#     sk_model=clf,
#     path="svc",
#     serialization_format="cloudpickle",
#     conda_env=conda_env,
#     signature=signature,
#     input_example=input_example,
# )

# MLFlow on Server

# import mlflow
#
# with mlflow.start_run():
#     mlflow.log_artifact("svc/")
from functools import partial

import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func
from kfp.dsl import pipeline


@partial(
    create_component_from_func,
    packages_to_install=["pandas", "scikit-learn"],
)
def load_gan_data( # gan data path
        data_path: OutputPath("csv"),
        target_path: OutputPath("csv"),
):

    import pandas as pd
    from sklearn.datasets import load_iris

    # minio 접근?
    import os
    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    # minio내 mlflow 폴더

    iris = load_iris()

    data = pd.DataFrame(iris["data"], columns=iris["feature_names"])
    target = pd.DataFrame(iris["target"], columns=["target"])

    data.to_csv(data_path, index=False)
    target.to_csv(target_path, index=False)


@partial(
    create_component_from_func,
    packages_to_install=["dill", "pandas", "scikit-learn", "mlflow"],
)
def train_from_csv(
        train_data_path: InputPath("csv"),
        train_target_path: InputPath("csv"),
        model_path: OutputPath("dill"),
        input_example_path: OutputPath("dill"),
        signature_path: OutputPath("dill"),
        conda_env_path: OutputPath("dill"),
        kernel: str,
):
    import dill
    import pandas as pd
    from sklearn.svm import SVC

    from mlflow.models.signature import infer_signature
    from mlflow.utils.environment import _mlflow_conda_env

    train_data = pd.read_csv(train_data_path)
    train_target = pd.read_csv(train_target_path)

    clf = SVC(kernel=kernel)
    clf.fit(train_data, train_target)

    with open(model_path, mode="wb") as file_writer:
        dill.dump(clf, file_writer)

    input_example = train_data.sample(1)
    with open(input_example_path, "wb") as file_writer:
        dill.dump(input_example, file_writer)

    signature = infer_signature(train_data, clf.predict(train_data))
    with open(signature_path, "wb") as file_writer:
        dill.dump(signature, file_writer)

    conda_env = _mlflow_conda_env(
        additional_pip_deps=["dill", "pandas", "scikit-learn"]
    )
    with open(conda_env_path, "wb") as file_writer:
        dill.dump(conda_env, file_writer)


@partial(
    create_component_from_func,
    packages_to_install=["dill", "pandas", "scikit-learn", "mlflow", "boto3"],
)
def upload_sklearn_model_to_mlflow(
        model_name: str,
        model_path: InputPath("dill"),
        input_example_path: InputPath("dill"),
        signature_path: InputPath("dill"),
        conda_env_path: InputPath("dill"),
):
    import os
    import dill
    from mlflow.sklearn import save_model
    from mlflow.tensorflow import load_model

    from mlflow.tracking.client import MlflowClient

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    with open(model_path, mode="rb") as file_reader:
        clf = dill.load(file_reader)

    with open(input_example_path, "rb") as file_reader:
        input_example = dill.load(file_reader)

    with open(signature_path, "rb") as file_reader:
        signature = dill.load(file_reader)

    with open(conda_env_path, "rb") as file_reader:
        conda_env = dill.load(file_reader)

    # 연결된 상태


    save_model(
        sk_model=clf,
        path=model_name,
        serialization_format="cloudpickle",
        conda_env=conda_env,
        signature=signature,
        input_example=input_example,
    )
    run = client.create_run(experiment_id="0")
    client.log_artifact(run.info.run_id, model_name)


@pipeline(name="mlflow_pipeline")
def mlflow_pipeline(kernel: str, model_name: str):
    iris_data = load_iris_data()
    model = train_from_csv(
        train_data=iris_data.outputs["data"],
        train_target=iris_data.outputs["target"],
        kernel=kernel,
    )
    _ = upload_sklearn_model_to_mlflow(
        model_name=model_name,
        model=model.outputs["model"],
        input_example=model.outputs["input_example"],
        signature=model.outputs["signature"],
        conda_env=model.outputs["conda_env"],
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlflow_pipeline, "mlflow_pipeline.yaml")