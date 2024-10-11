import numpy as np
import tensorflow as tf

import logging

from .batcher import Batcher


def compute_pcd_sse(x, y):
    squared_difference_per_point = tf.math.squared_difference(x, y)  # (?, 5023, 3)
    squared_difference_sum_per_point = tf.math.reduce_sum(squared_difference_per_point, axis=2)  # (?, 5023)
    total_squared_difference = tf.math.reduce_sum(squared_difference_sum_per_point)  # (?)
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


# TODO 用 dataclass decorator 簡化 constructor
class Model:
    def __init__(
        self,
        train_batcher: Batcher,
        val_batcher: Batcher,
        test_batcher: Batcher,
        learning_rate: float,
        epochs: int,
        validation_steps: int,
    ):
        self.train_batcher = train_batcher
        self.val_batcher = val_batcher
        self.test_batcher = test_batcher

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_steps = validation_steps

        # TODO 引入不同 optimizer (透過 config.yaml 判斷要用什麼)
        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
        )

        self.model = self.build_model()
        self.model.summary()

    def build_model(self) -> tf.keras.Model:
        subject_id_ont_hot_shape = (None,)
        deepspeech_feature_shape = (None, 16, 29)

        input_c = tf.keras.Input(shape=subject_id_ont_hot_shape[1:], name="input_c")
        input_x = tf.keras.Input(shape=deepspeech_feature_shape[1:], name="input_x")
        c = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")(input_c)  # (None, 8,)

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

        factor = 1.0  # TODO

        # 第一層: conv2d (None, 8, 1, 32)
        conv1 = build_conv_layer(name="conv1", x=x, filters=int(32 * factor))

        # 第二層: conv2d (None, 4, 1, 32)
        conv2 = build_conv_layer(name="conv2", x=conv1, filters=int(32 * factor))

        # 第三層: conv2d (None, 2, 1, 64)
        conv3 = build_conv_layer(name="conv3", x=conv2, filters=int(64 * factor))

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = build_conv_layer(name="conv4", x=conv3, filters=int(64 * factor))

        x = tf.keras.layers.Flatten()(conv4)

        # 第二次　Identity concat　加上　subject_id_ont_hot
        x = tf.keras.layers.Concatenate(axis=-1, name="id_concat2")([x, c])

        fc1 = tf.keras.layers.Dense(units=128, activation="tanh", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(units=50, activation=None, name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(units=5023 * 3, activation=None, name="fc3")(fc2)  # (?, 5023 * 3)

        y = tf.keras.layers.Reshape((5023, 3), name="delta_pcd")(fc3)  # (?, 5023, 3)

        return tf.keras.Model(inputs=[input_c, input_x], outputs=[y])

    def position_loss(self, true_pcd, pred_pcd):
        return compute_pcd_sse(true_pcd, pred_pcd)

    def velocity_loss(self, true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr):
        return compute_pcd_sse(true_pcd_curr - true_pcd_prev, pred_pcd_curr - pred_pcd_prev)

    def train(self):

        train_loss_metric = tf.keras.metrics.Mean(name="train_loss")
        val_loss_metric = tf.keras.metrics.Mean(name="val_loss")

        train_steps = self.train_batcher.get_num_batches()
        val_steps = 100  # self.train_batcher.get_num_batches()

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}/{self.epochs}")
            train_progbar = tf.keras.utils.Progbar(target=train_steps)
            val_progbar = tf.keras.utils.Progbar(target=val_steps)

            for step in range(train_steps):
                subject_id, template_pcd, true_pcd, audio = self.train_batcher.get_next()

                true_pcd_prev = true_pcd[..., 0]
                true_pcd_curr = true_pcd[..., 1]

                audio_prev = audio[..., 0]
                audio_curr = audio[..., 1]

                with tf.GradientTape() as tape:
                    # TODO 是否要 training=True? (目前看起來不用, 有時間開起來看有沒有什麼問題)
                    pred_pcd_prev = template_pcd + self.model([subject_id, audio_prev], training=False)
                    pred_pcd_curr = template_pcd + self.model([subject_id, audio_curr], training=True)

                    loss = self.position_loss(true_pcd_curr, pred_pcd_curr) + 10.0 * self.velocity_loss(
                        true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr
                    )

                gradients = tape.gradient(loss, self.model.trainable_variables)  # 計算梯度
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))  # 更新權重

                train_loss_metric.update_state(loss)
                train_progbar.update(step + 1, values=[("loss", train_loss_metric.result())])

            # Validation Loop
            for step in range(val_steps):
                subject_id, template_pcd, true_pcd, audio = self.train_batcher.get_next()

                true_pcd_prev = true_pcd[..., 0]
                true_pcd_curr = true_pcd[..., 1]

                audio_prev = audio[..., 0]
                audio_curr = audio[..., 1]

                pred_pcd_prev = template_pcd + self.model([subject_id, audio_prev], training=False)
                pred_pcd_curr = template_pcd + self.model([subject_id, audio_curr], training=False)

                val_loss = self.position_loss(true_pcd_curr, pred_pcd_curr) + 10.0 * self.velocity_loss(
                    true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr
                )

                val_loss_metric.update_state(val_loss)
                val_progbar.update(step + 1, values=[("loss", train_loss_metric.result())])

            print(f"平均 SSE (train): {train_loss_metric.result()}")
            print(f"平均 SSE (val): {val_loss_metric.result()}")
            print()
            train_loss_metric.reset_states()
            val_loss_metric.reset_states()

    # TODO
    def eval(self):
        pass

    def save(self, dir_path: str = "models/"):
        self.model.save(dir_path, overwrite=False)
