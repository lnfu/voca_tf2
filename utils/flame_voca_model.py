import tensorflow as tf

import logging

from .flame import Flame
from .common import log_execution
from .voca_model import VocaModel, build_conv_layer


class ConcatShapeLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ConcatShapeLayer, self).__init__()
        self.s = tf.Variable(
            initial_value=tf.random.normal(shape=(300,), dtype=tf.float32),
            trainable=True,
            name="flame_shape_params",
        )

    def call(self, inputs):
        return tf.concat(
            [
                tf.repeat(
                    tf.expand_dims(self.s, axis=0), repeats=tf.shape(inputs)[0], axis=0
                ),  # (300,) -> (1, 300) -> (?, 300)
                inputs,  # (?, 115)
            ],
            axis=1,
        )  # (?, 415)


class FlameVocaModel(VocaModel):

    def build_model(self) -> tf.keras.Model:
        deepspeech_feature_shape = (None, 16, 29)

        input_x = tf.keras.Input(shape=deepspeech_feature_shape[1:], name="input_x")

        # TODO 是否需要 translation?

        # Batch Normalization
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(input_x)

        x = tf.reshape(x, (-1, 16, 1, 29))  # (None, 16, 1, 29)

        # 第一層: conv2d (None, 8, 1, 32)
        conv1 = build_conv_layer(name="conv1", x=x, filters=int(32 * self.factor))

        # 第二層: conv2d (None, 4, 1, 32)
        conv2 = build_conv_layer(name="conv2", x=conv1, filters=int(32 * self.factor))

        # 第三層: conv2d (None, 2, 1, 64)
        conv3 = build_conv_layer(name="conv3", x=conv2, filters=int(64 * self.factor))

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = build_conv_layer(name="conv4", x=conv3, filters=int(64 * self.factor))

        x = tf.keras.layers.Flatten()(conv4)

        fc1 = tf.keras.layers.Dense(units=128, activation="tanh", name="fc1")(x)
        fc2 = tf.keras.layers.Dense(units=50, activation=None, name="fc2")(fc1)
        fc3 = tf.keras.layers.Dense(units=115, activation=None, name="fc3")(
            fc2
        )  # (?, 115) FLAME expression (100) + pose (15) parameters

        y = ConcatShapeLayer()(fc3)

        return tf.keras.Model(inputs=[input_x], outputs=[y])

    def run_epoch(self, loss_metric, is_training=True):

        batcher = self.train_batcher if is_training else self.val_batcher
        steps = batcher.get_num_batches()
        progbar = tf.keras.utils.Progbar(target=steps)

        for step in range(steps):
            audio, true_pcd = batcher.get_next()

            with tf.GradientTape() as tape:
                pred_flame_params = self.model([audio], training=is_training)

                assert pred_flame_params.shape[1] == 415  # TODO

                pred_pcd = tf.map_fn(
                    lambda x: Flame.calculate_pcd_by_param(x), pred_flame_params
                )

                loss = self.position_loss(
                    true_pcd, pred_pcd
                ) + 10.0 * self.velocity_loss(true_pcd, pred_pcd) + 100.0 * self.acceleration_loss(true_pcd, pred_pcd)

            gradients = tape.gradient(loss, self.model.trainable_variables)  # 計算梯度
            self.optimizer.apply_gradients(
                zip(gradients, self.model.trainable_variables)
            )  # 更新權重

            loss_metric.update_state(loss)
            progbar.update(step + 1, values=[("loss", loss_metric.result())])

        logging.info(
            f"平均 SSE ({'train' if is_training else 'val'}): {loss_metric.result()}"
        )
