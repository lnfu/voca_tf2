import tensorflow as tf

import os
import logging

from .batcher import Batcher
from .common import log_execution
from tensorflow.python.summary.summary_iterator import summary_iterator
from datetime import datetime


@tf.function
def diff(arr):
    return arr[1:] - arr[:-1]


@tf.function
def compute_pcd_sse(x, y):
    squared_difference_per_point = tf.math.squared_difference(
        x, y)  # (?, 5023, 3)
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
    activation: str = "silu",
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
        loss_weights,
        factor: float = 1.0,
        optimizer: str = "Adam",
        beta_1: float = 0.9,
        reset: bool = False,
        checkpoint_dir_path: str = "checkpoints/",
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
        self.loss_weights = loss_weights

        # optimizer
        if optimizer.lower() == "adam":
            self.optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=beta_1,
            )
        elif optimizer.lower() == "sgd":
            self.optimizer = tf.keras.optimizers.SGD(
                learning_rate=self.learning_rate,
                clipnorm=1.
            )
        else:
            raise ValueError(f"不支援的 Optimizer: {optimizer}")

        # model
        self.model = self.build_model()
        self.model.summary()

        # checkpoint
        self.checkpoint_dir = checkpoint_dir_path
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.model
        )
        self.checkpoint_manager = tf.train.CheckpointManager(
            self.checkpoint, self.checkpoint_dir, max_to_keep=3
        )
        if not reset and self.checkpoint_manager.latest_checkpoint:
            logging.info(
                f"根據 checkpoint 繼續訓練: {self.checkpoint_manager.latest_checkpoint}"
            )
            self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        elif reset:
            logging.info("從頭開始訓練 (忽略 checkpoint)")

    def build_model(self) -> tf.keras.Model:
        subject_id_ont_hot_shape = (None,)
        deepspeech_feature_shape = (None, 16, 29)

        input_c = tf.keras.Input(
            shape=subject_id_ont_hot_shape[1:], name="input_c")
        input_x = tf.keras.Input(
            shape=deepspeech_feature_shape[1:], name="input_x")
        c = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")(
            input_c
        )  # (None, 8,)

        # Batch Normalization
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)(input_x)

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
        conv1 = build_conv_layer(
            name="conv1", x=x, filters=int(32 * self.factor))

        # 第二層: conv2d (None, 4, 1, 32)
        conv2 = build_conv_layer(
            name="conv2", x=conv1, filters=int(32 * self.factor))

        # 第三層: conv2d (None, 2, 1, 64)
        conv3 = build_conv_layer(
            name="conv3", x=conv2, filters=int(64 * self.factor))

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = build_conv_layer(
            name="conv4", x=conv3, filters=int(64 * self.factor))

        x = tf.keras.layers.Flatten()(conv4)

        # 第二次　Identity concat　加上　subject_id_ont_hot
        x = tf.keras.layers.Concatenate(axis=-1, name="id_concat2")([x, c])

        fc1 = tf.keras.layers.Dense(
            units=128, activation="tanh", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(units=50, activation=None, name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(units=5023 * 3, activation=None, name="fc3")(
            fc2
        )  # (?, 5023 * 3)

        y = tf.keras.layers.Reshape(
            (5023, 3), name="delta_pcd")(fc3)  # (?, 5023, 3)

        return tf.keras.Model(inputs=[input_c, input_x], outputs=[y])

    @tf.function
    def position_loss(self, true_pcd, pred_pcd):
        return compute_pcd_sse(true_pcd, pred_pcd)

    @tf.function
    def velocity_loss(self, true_pcd, pred_pcd):
        true_pcd_velocity = diff(true_pcd)
        pred_pcd_velocity = diff(pred_pcd)
        return compute_pcd_sse(true_pcd_velocity, pred_pcd_velocity)

    @tf.function
    def acceleration_loss(self, true_pcd, pred_pcd):
        true_pcd_velocity = diff(true_pcd)
        pred_pcd_velocity = diff(pred_pcd)

        true_pcd_acceleration = diff(true_pcd_velocity)
        pred_pcd_acceleration = diff(pred_pcd_velocity)

        return compute_pcd_sse(true_pcd_acceleration, pred_pcd_acceleration)

    def run_epoch(self, loss_metric, is_training=True):
        raise NotImplementedError

    def get_last_logged_epoch(self, log_dir: str):
        """
        用來獲取上一次訓練到第幾個 iteration (epoch)
        以便讓相同 train_tag 下的 training 可以接續紀錄
        """
        latest_epoch = 0

        if not os.path.exists(log_dir):
            return latest_epoch

        for root, dirs, files in os.walk(log_dir):
            for file in files:
                if "events.out.tfevents" in file:
                    event_file = os.path.join(root, file)
                    for summary in summary_iterator(event_file):
                        for value in summary.summary.value:
                            if value.tag == "train_loss":
                                latest_epoch = max(latest_epoch, summary.step)
        return latest_epoch

    @log_execution
    def train(self):
        """
        訓練模型
        """

        self.train_tag = datetime.now().strftime(
            '%Y_%m_%d_%H_%M_%S')

        latest_epoch = self.get_last_logged_epoch(
            str(os.path.join("logs/", f"train_{self.train_tag}")))

        train_summary_writer = tf.summary.create_file_writer(
            os.path.join("logs/", f"train_{self.train_tag}")
        )
        val_summary_writer = tf.summary.create_file_writer(
            os.path.join("logs/", f"val_{self.train_tag}"))

        train_loss_metric = tf.keras.metrics.Mean(name="train_loss_metric")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss_metric")

        for epoch in range(self.epochs):
            logging.info(f"Epoch {epoch+1}/{self.epochs}")

            self.run_epoch(train_loss_metric, is_training=True)
            with train_summary_writer.as_default():
                tf.summary.scalar(
                    "loss", train_loss_metric.result(), step=epoch + latest_epoch + 1)

            if self.validation_freq != 0 and epoch % self.validation_freq == 0:
                self.run_epoch(val_loss_metric, is_training=False)
                with val_summary_writer.as_default():
                    tf.summary.scalar(
                        "loss", val_loss_metric.result(), step=epoch + latest_epoch + 1)

            train_loss_metric.reset_states()
            val_loss_metric.reset_states()
            self.checkpoint_manager.save()

            logging.info("")

    def save(self, dir_path: str = "models/"):
        """
        儲存模型
        """

        self.model.save(
            os.path.join(dir_path, self.train_tag),
            overwrite=False
        )
        return self.train_tag

    # TODO test model
    def eval(self):
        """
        測試模型
        """
        pass
