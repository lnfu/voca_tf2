import tensorflow as tf

import logging

from .flame import Flame
from .voca_model import VocaModel, build_conv_layer


class ConcatShapeLayer(tf.keras.layers.Layer):
    def __init__(self, num_subjects=12):
        super(ConcatShapeLayer, self).__init__()
        self.s = tf.Variable(
            initial_value=tf.random.normal(
                shape=(num_subjects, 300), dtype=tf.float32),  # (12, 300)
            trainable=True,
            name="all_subjects_flame_shape_params",
        )

    def call(self, inputs, subject_id=0):
        return tf.concat(
            [
                tf.gather(self.s, subject_id),  # (?, 300)
                inputs,  # (?, 115)
            ],
            axis=1,
        )  # (?, 415)


class FlameVocaModel(VocaModel):

    def build_model(self) -> tf.keras.Model:
        deepspeech_feature_shape = (None, 16, 29)
        subject_id_shape = (None,)

        input_x = tf.keras.Input(
            shape=deepspeech_feature_shape[1:],
            dtype=tf.float32,
            name="input_x"
        )

        input_id = tf.keras.Input(
            shape=subject_id_shape[1:],
            dtype=tf.int32,
            name="input_id"
        )

        # Batch Normalization
        x = tf.keras.layers.BatchNormalization(
            epsilon=1e-5, momentum=0.9)(input_x)

        x = tf.reshape(x, (-1, 16, 1, 29))  # (None, 16, 1, 29)

        # 第一層: conv2d (None, 8, 1, 64)
        conv1 = build_conv_layer(
            name="conv1", x=x, filters=int(64 * self.factor), activation="gelu")

        # 第二層: conv2d (None, 4, 1, 128)
        conv2 = build_conv_layer(
            name="conv2", x=conv1, filters=int(128 * self.factor), activation="gelu")

        # 第三層: conv2d (None, 2, 1, 256)
        conv3 = build_conv_layer(
            name="conv3", x=conv2, filters=int(256 * self.factor), activation="gelu")

        # 第四層: conv2d (None, 1, 1, 512)
        conv4 = build_conv_layer(
            name="conv4", x=conv3, filters=int(512 * self.factor), activation=None)

        x = tf.keras.layers.Flatten()(conv4)

        fc1 = tf.keras.layers.Dense(
            units=512, activation="gelu", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(
            units=256, activation="gelu", name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(
            units=128, activation="gelu", name="fc3")(fc2)
        fc4 = tf.keras.layers.Dense(units=115, activation=None, name="fc4")(
            fc3
        )  # (?, 115) FLAME expression (100) + pose (15) parameters

        y = ConcatShapeLayer()(fc4, input_id)

        return tf.keras.Model(inputs=[input_x, input_id], outputs=[y])

    def run_epoch(self, loss_metric, is_training=True):

        batcher = self.train_batcher if is_training else self.val_batcher
        steps = batcher.get_num_batches()
        progbar = tf.keras.utils.Progbar(target=steps)

        for step in range(steps):
            subject_id, audio, true_pcd = batcher.get_next()

            with tf.GradientTape() as tape:
                pred_flame_params = self.model(
                    [audio, subject_id], training=is_training)

                assert pred_flame_params.shape[1] == 415  # TODO

                pred_pcd = tf.map_fn(
                    lambda x: Flame.calculate_pcd_by_param(
                        x), pred_flame_params
                )

                loss = 0.0
                if self.loss_weights["pos"] != 0:
                    loss += self.loss_weights["pos"] * \
                        self.position_loss(true_pcd, pred_pcd)
                if self.loss_weights["vel"] != 0:
                    loss += self.loss_weights["vel"] * \
                        self.velocity_loss(true_pcd, pred_pcd)
                if self.loss_weights["acc"] != 0:
                    loss += self.loss_weights["acc"] * \
                        self.acceleration_loss(true_pcd, pred_pcd)

                if is_training:
                    gradients = tape.gradient(
                        loss, self.model.trainable_variables)  # 計算梯度
                    self.optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )  # 更新權重

            loss_metric.update_state(loss)
            progbar.update(step + 1, values=[("loss", loss_metric.result())])

        logging.info(
            f"平均 SSE ({'train' if is_training else 'val'}): {loss_metric.result()}"
        )
