# TODO refactor: conv2d 抽出來成一個 method

import numpy as np
import tensorflow as tf

import logging

from .batcher import Batcher


# TODO refactor: rename function name


def compute_pcd_squared_difference(x, y):
    squared_difference_per_point = tf.math.squared_difference(x, y)  # (?, 5023, 3)
    squared_difference_sum_per_point = tf.math.reduce_sum(squared_difference_per_point, axis=2)  # (?, 5023)
    total_squared_difference = tf.math.reduce_sum(squared_difference_sum_per_point)  # (?)
    return total_squared_difference


def custom_loss(y_true, y_pred):  # (?, 5023, 3)
    # y_true = (None, 5023, 3, 2) --> template + gt

    pred_pcd_prev = y_pred[..., 0] + y_true[..., 0]  # (None, 5023, 3)
    pred_pcd_current = y_pred[..., 1] + y_true[..., 0]  # (None, 5023, 3)

    true_pcd_prev = y_true[..., 1]  # (None, 5023, 3)
    true_pcd_current = y_true[..., 2]  # (None, 5023, 3)

    position_squared_difference = compute_pcd_squared_difference(pred_pcd_current, true_pcd_current)

    pred_pcd_diff = pred_pcd_current - pred_pcd_prev  # (None, 5023, 3)
    true_pcd_diff = true_pcd_current - true_pcd_prev  # (None, 5023, 3)

    velocity_squared_difference = compute_pcd_squared_difference(pred_pcd_diff, true_pcd_diff)

    return position_squared_difference + velocity_squared_difference


# TODO 用 dataclass decorator 簡化 constructor
class Model:
    def __init__(
        self,
        train_batcher: Batcher,  #
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

        # x = tf.keras.layers.Reshape((16, 1, 29))(x)  # (None, 16, 1, 29) # TODO remove
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
        conv1 = tf.keras.layers.Conv2D(
            filters=int(32 * factor),
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            activation="relu",
            name="conv1",
        )(x)

        # 第二層: conv2d (None, 4, 1, 32)
        conv2 = tf.keras.layers.Conv2D(
            filters=int(32 * factor),
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            activation="relu",
            name="conv2",
        )(conv1)

        # 第三層: conv2d (None, 2, 1, 64)
        conv3 = tf.keras.layers.Conv2D(
            filters=int(64 * factor),
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            activation="relu",
            name="conv3",
        )(conv2)

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = tf.keras.layers.Conv2D(
            filters=int(64 * factor),
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            activation="relu",
            name="conv4",
        )(conv3)

        x = tf.keras.layers.Flatten()(conv4)

        # 第二次　Identity concat　加上　subject_id_ont_hot
        x = tf.keras.layers.Concatenate(axis=-1, name="id_concat2")([x, c])

        fc1 = tf.keras.layers.Dense(units=128, activation="tanh", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(units=50, activation=None, name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(units=5023 * 3, activation=None, name="fc3")(fc2)  # (?, 5023 * 3)

        y = tf.keras.layers.Reshape((5023, 3), name="delta_pcd")(fc3)  # (?, 5023, 3)

        return tf.keras.Model(inputs=[input_c, input_x], outputs=[y])

    # TODO 可能改成一個 epoch 只能跑完每個 batch 就不能重複了?
    def data_generator(self, batcher: Batcher):
        while True:
            batch_subject_id, batch_template_pcd, batch_pcd, batch_audio = batcher.get_next()
            yield (batch_subject_id, batch_audio), np.concatenate(
                (np.expand_dims(batch_template_pcd, axis=-1), batch_pcd), axis=-1
            )

    def create_dataset(self, generator):
        return tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                (
                    tf.TensorSpec(shape=(None,), dtype=tf.float32),  # subject id
                    tf.TensorSpec(shape=(None, 16, 29, 2), dtype=tf.float32),  # audio (DeepSpeech)
                ),
                tf.TensorSpec(shape=(None, 5023, 3, 3), dtype=tf.float32),  # template pcd & ground truth pcd
            ),
        )

    def position_loss(self, true_pcd, pred_pcd):
        return compute_pcd_squared_difference(true_pcd, pred_pcd)

    def velocity_loss(self, true_pcd_prev, pred_pcd_prev, true_pcd_curr, pred_pcd_curr):
        return compute_pcd_squared_difference(true_pcd_curr - true_pcd_prev, pred_pcd_curr - pred_pcd_prev)

    def train(self):

        # TODO remove
        # train_dataset = self.create_dataset(lambda: self.data_generator(self.train_batcher))
        # val_dataset = self.create_dataset(lambda: self.data_generator(self.val_batcher))

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
                    # TODO 是否要 training=True?
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

    def eval(self):
        # TODO
        # test_dataset = self.create_dataset(lambda: self.data_generator(self.test_batcher))
        # test_loss = self.model.evaluate(test_dataset, steps=self.test_batcher.get_num_batches())
        # print("Test Loss:", test_loss)
        pass

    def save(self, dir_path: str = "models/"):
        self.model.save(dir_path, overwrite=False)
