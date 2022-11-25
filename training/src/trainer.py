
import argparse
import json
import os

from utils import create_model, prepare_dataset, configure_keras_callbacks
import tensorflow as tf


def get_args():
    """Parses args. Must include all hyperparameters you want to tune."""

    def custom_int(value):
        return int(float(value))

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset-uri", type=str)
    parser.add_argument("--num-classes", type=custom_int)
    parser.add_argument("--compression", type=str, default="NONE")
    parser.add_argument("--base-model-dir", type=str, default=None)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--image-size", type=json.loads, default=[124, 124])
    parser.add_argument("--epochs", type=custom_int, default=1)
    parser.add_argument("--steps-per-epoch", type=custom_int, default=100)
    parser.add_argument("--validation-steps", type=custom_int, default=100)
    parser.add_argument("--batch-size", type=custom_int, default=1)
    parser.add_argument("--shuffle", type=bool, default=True)
    parser.add_argument("--shuffle-buffer", type=custom_int, default=64)
    parser.add_argument("--seed", type=custom_int, default=None)
    parser.add_argument("--save-checkpoints", type=bool, default=False)
    parser.add_argument("--checkpoint-kwargs", type=json.loads, default={})
    parser.add_argument("--tensorboard-log-root", type=str, default=None)
    parser.add_argument("--tensorboard-kwargs", type=json.loads, default={})

    parser.add_argument("--hypertune", type=bool, default=False)

    # tunable parameters
    parser.add_argument("--num-neurons", type=custom_int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--learning-rate", type=float, default=0.001)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    print(args)
    print(os.environ)

    if args.tensorboard_log_root:
        job_subdir = os.environ.get("CLOUD_ML_JOB_ID", "")
        trial_subdir = os.environ.get("CLOUD_ML_TRIAL_ID", "")
        tensorboard_log_dir = os.path.join(
            args.tensorboard_log_root, job_subdir, trial_subdir
        ).rstrip("/")
    else:
        tensorboard_log_dir = None

    checkpoints_dir = os.environ.get("AIP_CHECKPOINT_DIR") if args.save_checkpoints else None

    trained_model_dir = os.environ.get("AIP_MODEL_DIR")

    print(tensorboard_log_dir, checkpoints_dir, trained_model_dir)

    model = create_model(
        image_size=args.image_size,
        num_classes=args.num_classes,
        base_model_dir=args.base_model_dir,
        num_neurons=args.num_neurons,
        dropout=args.dropout,
        activation=args.activation,
        learning_rate=args.learning_rate,
    )

    callbacks = configure_keras_callbacks(
        tensorboard_log_dir=tensorboard_log_dir,
        tensorboard_kwargs=args.tensorboard_kwargs,
        checkpoints_dir=checkpoints_dir,
        checkpoint_kwargs=args.checkpoint_kwargs,
        hypertune=args.hypertune,
        hypertune_kwargs=dict(metric=os.environ.get("CLOUD_ML_HP_METRIC_TAG")),
    )

    train_data, val_data = prepare_dataset(
        train_dataset_uri=args.train_dataset_uri,
        compression=args.compression,
        shuffle=args.shuffle,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
        batch_size=args.batch_size,
        val_size=args.val_size,
    )

    model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=val_data,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
    )

    metric_values = model.evaluate(val_data)

    metrics = {k: v for k, v in zip(model.metrics_names, metric_values)}

    metadata = dict(framework=f"Tensorflow {tf.__version__}")

    model.save(trained_model_dir)

    trained_model_dir_fuse = trained_model_dir.replace("gs://", "/gcs/")

    with open(os.path.join(trained_model_dir_fuse, "metrics.json"), "w") as fh:
        json.dump(metrics, fh)
    with open(os.path.join(trained_model_dir_fuse, "metadata.json"), "w") as fh:
        json.dump(metadata, fh)
