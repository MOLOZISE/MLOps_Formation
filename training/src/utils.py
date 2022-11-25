import os
import pickle  # nosec
import shutil
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, Input, Resizing
from tensorflow.keras.models import Model

import tensorflow_hub as hub

import hypertune


class HyperTuneCallback(tf.keras.callbacks.Callback):
    def __init__(self, metric=None) -> None:
        super().__init__()
        self.metric = metric
        self.hpt = hypertune.HyperTune()

    def on_epoch_end(self, epoch, logs=None):
        if logs and self.metric in logs:
            self.hpt.report_hyperparameter_tuning_metric(
                hyperparameter_metric_tag=self.metric,
                metric_value=logs[self.metric],
                global_step=epoch,
            )


def prepare_dataset(
    train_dataset_uri, compression, shuffle, shuffle_buffer, seed, batch_size, val_size
):
    with open(
        os.path.join(train_dataset_uri.replace("gs://", "/gcs/"), "element_spec.pickle"), "rb"
    ) as fh:
        element_spec = pickle.load(fh)  # nosec

    dataset = tf.data.experimental.load(
        train_dataset_uri, element_spec=element_spec, compression=compression
    )

    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, seed=seed)

    dataset_size = dataset.cardinality().numpy()
    num_train_samples = (1.0 - val_size) * dataset_size

    train_data, val_data = dataset.take(num_train_samples), dataset.skip(num_train_samples)

    if shuffle:
        train_data = train_data.shuffle(shuffle_buffer, seed=seed)
        val_data = val_data.shuffle(shuffle_buffer, seed=seed)

    train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    val_data = val_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return train_data, val_data


def create_model(
    image_size, num_classes, base_model_dir, num_neurons,
    dropout, activation, learning_rate
):
    inputs = Input(shape=(None, None, 3))
    x = Resizing(*image_size)(inputs)

    base_model = None
    if base_model_dir.startswith("gs://"):
        base_model = tf.keras.models.load_model(base_model_dir.replace("gs://", "/gcs/"))

    if base_model_dir.startswith("https://tfhub.dev"):
        base_model = hub.KerasLayer(
          base_model_dir,
          trainable=False,
          input_shape=(*image_size, 3)
        )

    if not base_model:
        base_model = MobileNetV2(weights="imagenet", input_tensor=x)
        base_model.trainable = False
        outputs = base_model.layers[-2].output
    else:
        outputs = base_model(x)

    model = Model(inputs=inputs, outputs=outputs)

    outputs = Dense(num_neurons, activation=activation)(model.output)
    outputs = Dropout(dropout)(outputs)
    outputs = Dense(num_classes, activation="softmax")(outputs)
    model = Model(inputs=model.input, outputs=outputs)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def configure_keras_callbacks(
    tensorboard_log_dir,
    tensorboard_kwargs,
    checkpoints_dir,
    checkpoint_kwargs,
    hypertune,
    hypertune_kwargs,
):
    callbacks = []

    if hypertune:
        callbacks.append(HyperTuneCallback(**hypertune_kwargs))

    if tensorboard_log_dir:
        fuse_path = tensorboard_log_dir.replace("gs://", "/gcs/")
        if Path(fuse_path).exists():
            shutil.rmtree(fuse_path)
        callbacks.append(
            tf.keras.callbacks.TensorBoard(log_dir=tensorboard_log_dir, **tensorboard_kwargs)
        )
    if checkpoints_dir:
        callbacks.append(tf.keras.callbacks.ModelCheckpoint(checkpoints_dir, **checkpoint_kwargs))

    return callbacks
