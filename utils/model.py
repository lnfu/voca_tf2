# TODO refactor: conv2d 抽出來成一個 method

import tensorflow as tf

from .batcher import Batcher


# TODO refactor: rename function name
def custom_loss(y_true, y_pred):  # (?, 5023, 3)
    squared_diff = tf.math.squared_difference(y_pred, y_true)  # (?, 5023, 3)
    error_sum = tf.math.reduce_sum(squared_diff, axis=2)  # (?, 5023)
    mean_error = tf.math.reduce_mean(error_sum)
    return mean_error


class Model:
    def __init__(
        self, batcher: Batcher, learning_rate: float, epochs: int, validation_steps: int
    ):
        self.batcher = batcher
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.validation_steps = validation_steps

    def build_model(self):
        subject_id_ont_hot_shape = (None,)
        template_pcd_shape = (None, 5023, 3)
        deepspeech_feature_shape = (None, 16, 29)

        input_c = tf.keras.Input(shape=subject_id_ont_hot_shape[1:], name="input_c")
        input_t = tf.keras.Input(shape=template_pcd_shape[1:], name="input_t")
        input_x = tf.keras.Input(shape=deepspeech_feature_shape[1:], name="input_x")
        c = tf.keras.layers.CategoryEncoding(num_tokens=8, output_mode="one_hot")(
            input_c
        )  # (None, 8,)

        # Batch Normalization
        x = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(input_x)

        # x = tf.keras.layers.Reshape((16, 1, 29))(x)  # (None, 16, 1, 29) # TODO remove
        x = tf.reshape(x, (-1, 16, 1, 29)) # (None, 16, 1, 29)

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
            filters=int(32 * factor),
            kernel_size=(3, 1),
            strides=(2, 1),
            padding="same",
            activation="relu",
            name="conv3",
        )(conv2)

        # 第四層: conv2d (None, 1, 1, 64)
        conv4 = tf.keras.layers.Conv2D(
            filters=int(32 * factor),
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
        fc3 = tf.keras.layers.Dense(units=5023 * 3, activation=None, name="fc3")(fc2) # (?, 5023 * 3)

        x = tf.keras.layers.Reshape((5023, 3))(fc3) # (?, 5023, 3)

        y = tf.keras.layers.Add(name="add")([x, input_t])

        return tf.keras.Model(inputs=[input_c, input_t, input_x], outputs=y)

    def train(self):

        model = self.build_model()

        def data_generator(split):
            self.batcher.set_split(split)
            while True:
                subject_ids, template_pcds, audios, label_pcds = self.batcher.get_next()
                yield (subject_ids, template_pcds, audios), label_pcds

        dataset = {}
        for split in {"train", "val", "test"}:
            dataset[split] = tf.data.Dataset.from_generator(
                lambda: data_generator(split=split),
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None,), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 5023, 3), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, 16, 29), dtype=tf.float32),
                    ),
                    tf.TensorSpec(
                        shape=(None, 5023, 3), dtype=tf.float32
                    ),  # ground_truth
                ),
            )

        model.summary()

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
        )

        model.compile(optimizer=optimizer, loss=custom_loss)

        history = model.fit(
            dataset["train"],
            steps_per_epoch=self.batcher.get_num_window("train")
            // self.batcher.batch_size,
            epochs=self.epochs,
            validation_data=dataset["val"],
            validation_steps=self.validation_steps,  # TODO 計算上限　(不然賄選到重複的 val)
        )

        test_loss = model.evaluate(dataset["test"], steps=100)
        print("Test Loss:", test_loss)

        model.save("models/")
