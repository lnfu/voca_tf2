import numpy as np
import tensorflow as tf

import os
import time
import logging

from .batcher import Batcher
from .common import log_execution


def compute_pcd_sse(x, y):
    squared_difference_per_point = tf.math.squared_difference(x, y)  # (?, 5023, 3)
    squared_difference_sum_per_point = tf.math.reduce_sum(
        squared_difference_per_point, axis=2
    )  # (?, 5023)
    total_squared_difference = tf.math.reduce_sum(
        squared_difference_sum_per_point
    )  # (?)
    return total_squared_difference


def build_conv_layer(
    name: str,
    x,
    filters: int,
    kernel_size=(3, 1),
    strides=(2, 1),
    padding: str = "same",
    activation: str = "relu",
):
    return tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=activation,
        name=name,
    )(x)


class VocaModel:
    def __init__(
        self,
        train_batcher: Batcher,
        val_batcher: Batcher,
        test_batcher: Batcher,
        learning_rate: float,
        epochs: int,
        validation_freq: int,
        factor: float = 1.0,
        optimizer: str = "Adam",
        beta_1: float = 0.9,
        reset: bool = False,
        checkpoint_dir: str = "checkpoints/",  # TODO config.yaml
    ):
        # batcher
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self.test_batcher = test_batcher

        # training parameters
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_freq = validation_freq
        self.factor = factor

        # optimizer
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=beta_1,
            )
        else:
            raise ValueError(f"不支援的 Optimizer: {optimizer}")

        logging.info("Initialized Model with the following hyperparameters:")
        logging.info(f"Learning Rate: {self.learning_rate}")
        logging.info(f"Epochs: {self.epochs}")
        logging.info(f"Validation Frequency: {self.validation_freq}")
        logging.info(f"Factor: {self.factor}")
        logging.info(f"Optimizer: {self.optimizer}")

        # model
        self.model = self.build_model()
        self.model.summary()

        # checkpoint
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=3
        )
        if not reset and self.checkpoint_manager.latest_checkpoint:
            logging.info(
                f"Restoring from checkpoint: {self.checkpoint_manager.latest_checkpoint}"
            )
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        elif reset:
            logging.info("Starting fresh training, ignoring checkpoints")

    def build_model(self) -> tf.keras.Model:
        subject_id_ont_hot_shape = (None,)
        deepspeech_feature_shape = (None, 16, 29)

        input_c = tf.keras.Input(shape=subject_id_ont_hot_shape[1:], name="input_c")
        input_x = tf.keras.Input(shape=deepspeech_feature_shape[1:], name="input_x")
        c = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")(
            input_c
        )  # (None, 8,)

        # Batch Normalization
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(input_x)

        x = tf.reshape(x, (-1, 16, 1, 29))  # (None, 16, 1, 29)

        # 第一次　Identity concat　加上　subject_id_ont_hot
        x = tf.keras.layers.Concatenate(axis=-1, name="id_concat1")(
            [
                x,
                tf.keras.layers.Lambda(lambda x: tf.tile(x, [1, 16, 1, 1]))(
                    tf.keras.layers.Reshape((1, 1, 8))(c)  # (None, 1, 8)
                ),  # (None, 16, 1, 8)
            ]
        )

        # 第一層: conv2d (None, 8, 1, 32)
        conv1 = build_conv_layer(name="conv1", x=x, filters=int(32 * self.factor))

        # 第二層: conv2d (None, 4, 1, 32)
        conv2 = build_conv_layer(name="conv2", x=conv1, filters=int(32 * self.factor))

        # 第三層: conv2d (None, 2, 1, 64)
        conv3 = build_conv_layer(name="conv3", x=conv2, filters=int(64 * self.factor))

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = build_conv_layer(name="conv4", x=conv3, filters=int(64 * self.factor))

        x = tf.keras.layers.Flatten()(conv4)

        # 第二次　Identity concat　加上　subject_id_ont_hot
        x = tf.keras.layers.Concatenate(axis=-1, name="id_concat2")([x, c])

        fc1 = tf.keras.layers.Dense(units=128, activation="tanh", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(units=50, activation=None, name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(units=5023 * 3, activation=None, name="fc3")(
            fc2
        )  # (?, 5023 * 3)

        y = tf.keras.layers.Reshape((5023, 3), name="delta_pcd")(fc3)  # (?, 5023, 3)

        return tf.keras.Model(inputs=[input_c, input_x], outputs=[y])

    def position_loss(self, true_pcd, pred_pcd):
        return compute_pcd_sse(true_pcd, pred_pcd)

    def velocity_loss(self, true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr):
        return compute_pcd_sse(
            true_pcd_curr - true_pcd_prev, pred_pcd_curr - pred_pcd_prev
        )

    def run_epoch(self, loss_metric, is_training=True):

        batcher = self.train_batcher if is_training else self.val_batcher
        steps = batcher.get_num_batches()
        progbar = tf.keras.utils.Progbar(target=steps)

        for step in range(steps):
            subject_id, template_pcd, true_pcd, audio = batcher.get_next()

            true_pcd_prev = true_pcd[..., 0]
            true_pcd_curr = true_pcd[..., 1]
            audio_prev = audio[..., 0]
            audio_curr = audio[..., 1]

            with tf.GradientTape() as tape:
                # TODO 是否要 training=True? (目前看起來不用, 有時間開起來看有沒有什麼問題)
                pred_pcd_prev = template_pcd + self.model(
                    [subject_id, audio_prev], training=False
                )
                pred_pcd_curr = template_pcd + self.model(
                    [subject_id, audio_curr], training=is_training
                )

                loss = self.position_loss(
                    true_pcd_curr, pred_pcd_curr
                ) + 10.0 * self.velocity_loss(
                    true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr
                )

            gradients = tape.gradient(loss, self.model.trainable_variables)  # 計算梯度
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )  # 更新權重

            loss_metric.update_state(loss)
            progbar.update(step + 1, values=[("loss", loss_metric.result())])

        logging.info(
            f"平均 SSE ({'train' if is_training else 'val'}): {loss_metric.result()}"
        )

    @log_execution
    def train(self):
        logging.info(f"開始跑 {self.epochs} 個 epoch")

        train_summary_writer = tf.summary.create_file_writer(
            os.path.join("logs/", "train")
        )
        val_summary_writer = tf.summary.create_file_writer(os.path.join("logs/", "val"))

        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch+1}/{self.epochs}")

            self.run_epoch(train_loss_metric, is_training=True)
            with train_summary_writer.as_default():
                tf.summary.scalar("loss", train_loss_metric.result(), step=epoch)

            if (epoch + 1) % self.validation_freq == 0:
                self.run_epoch(val_loss_metric, is_training=False)
                with val_summary_writer.as_default():
                    tf.summary.scalar("loss", val_loss_metric.result(), step=epoch)

            train_loss_metric.reset_states()
            val_loss_metric.reset_states()
            self.checkpoint_manager.save()

            logging.info("")

    def save(self, dir_path: str = "models/"):
        self.model.save(dir_path, overwrite=False)

    # TODO
    def eval(self):
        pass
