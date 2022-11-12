from functools import partial

import kfp
from kfp.components import InputPath, OutputPath, create_component_from_func
from kfp.dsl import pipeline


@partial(
    create_component_from_func,
    base_image="ghcr.io/molozise/kogpt2:latest",
)
def kogpt2_app(
):
    import torch
    import string
    import streamlit as st
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    @st.cache(allow_output_mutation=True)
    def get_model():
        model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
        model.eval()
        return model

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>',
                                                        mask_token='<mask>')

    default_text = "korean is always busy"

    N_SENT = 3

    model = get_model()
    st.title("KoGPT2 Demo Page(ver 2.0)")

    st.markdown("""
    ### model
    | Model       |  # of params |   Type   | # of layers  | # of heads | ffn_dim | hidden_dims | 
    |--------------|:----:|:-------:|--------:|--------:|--------:|--------------:|
    | `KoGPT2` |  125M  |  Decoder |   12     | 12      | 3072    | 768 | 
    ### sampling method
    - greedy sampling
    - max out length : 128/1,024
    ## Conditional Generation
    """)

    text = st.text_area("Input Text:", value=default_text)
    st.write(text)
    punct = ('!', '?', '.')

    if text:
        st.markdown("## Predict")
        with st.spinner('processing..'):
            print(f'input > {text}')
            input_ids = tokenizer(text)['input_ids']
            gen_ids = model.generate(torch.tensor([input_ids]),
                                     max_length=128,
                                     repetition_penalty=2.0)
            generated = tokenizer.decode(gen_ids[0, :].tolist()).strip()
            if generated != '' and generated[-1] not in punct:
                for i in reversed(range(len(generated))):
                    if generated[i] in punct:
                        break
                generated = generated[:(i + 1)]
            print(f'KoGPT > {generated}')
        st.write(generated)

@partial(
    create_component_from_func,
    base_image="ghcr.io/molozise/kogpt2:latest",
)
def kogpt2_upload_config(
        model_path: OutputPath("dill"),
        input_example_path: OutputPath("dill"),
        signature_path: OutputPath("dill"),
        conda_env_path: OutputPath("dill"),
):
    import torch
    import string
    from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast

    import dill
    import pandas as pd

    from mlflow.models.signature import infer_signature
    from mlflow.utils.environment import _mlflow_conda_env

    tokenizer = PreTrainedTokenizerFast.from_pretrained("skt/kogpt2-base-v2",
                                                        bos_token='</s>',
                                                        eos_token='</s>',
                                                        unk_token='<unk>',
                                                        pad_token='<pad>',
                                                        mask_token='<mask>')

    default_text = "korean is always busy"

    N_SENT = 3
    punct = ('!', '?', '.')

    model = GPT2LMHeadModel.from_pretrained('skt/kogpt2-base-v2')
    model.eval()

    def generating(model, text):
        input_ids = tokenizer(text)['input_ids']
        gen_ids = model.generate(torch.tensor([input_ids]),
                                 max_length=128,
                                 repetition_penalty=2.0)
        generated = tokenizer.decode(gen_ids[0, :].tolist()).strip()
        if generated != '' and generated[-1] not in punct:
            for i in reversed(range(len(generated))):
                if generated[i] in punct:
                    break
            generated = generated[:(i + 1)]
        return generated

    with open(model_path, mode="wb") as file_writer:
        dill.dump(model, file_writer)

    input_example = default_text
    with open(input_example_path, "wb") as file_writer:
        dill.dump(input_example, file_writer)

    signature = infer_signature(default_text, generating(model, default_text))
    with open(signature_path, "wb") as file_writer:
        dill.dump(signature, file_writer)

    conda_env = _mlflow_conda_env(
        additional_pip_deps=["dill", "torch", "transformers"]
    )
    with open(conda_env_path, "wb") as file_writer:
        dill.dump(conda_env, file_writer)

@partial(
    create_component_from_func,
    base_image="ghcr.io/molozise/kogpt2:latest",
)
def kogpt2_upload_to_mlflow(
        model_name: str,
        model_path: InputPath("dill"),
        input_example_path: InputPath("dill"),
        signature_path: InputPath("dill"),
        conda_env_path: InputPath("dill"),
):

    import os
    import dill
    from mlflow.pytorch import save_model

    from mlflow.tracking.client import MlflowClient

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://minio-service.kubeflow.svc:9000"
    os.environ["AWS_ACCESS_KEY_ID"] = "minio"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"

    client = MlflowClient("http://mlflow-server-service.mlflow-system.svc:5000")

    with open(model_path, mode="rb") as file_reader:
        kogpt2 = dill.load(file_reader)

    with open(input_example_path, "rb") as file_reader:
        input_example = dill.load(file_reader)

    with open(signature_path, "rb") as file_reader:
        signature = dill.load(file_reader)

    with open(conda_env_path, "rb") as file_reader:
        conda_env = dill.load(file_reader)

    save_model(pytorch_model=kogpt2, path=model_name, conda_env=conda_env, signature=signature, pip_requirements=["torch", "transformers"])

    run = client.create_run(experiment_id="1")
    client.log_artifact(run.info.run_id, model_name)

@pipeline(name="mlflow_pipeline kogpt2")
def mlflow_pipeline(model_name: str):
    _ = kogpt2_app()
    model = kogpt2_upload_config()
    _ = kogpt2_upload_to_mlflow(
        model_name=model_name,
        model=model.outputs["model"],
        input_example=model.outputs["input_example"],
        signature=model.outputs["signature"],
        conda_env=model.outputs["conda_env"],
    )


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(mlflow_pipeline, "mlflow_pipeline_kogpt2.yaml")