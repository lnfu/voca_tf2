import numpy as np
import tensorflow as tf

import logging

from .common import log_execution


class Inference:
    def __init__(self, model_path: str):
        logging.info("正在載入 VOCA 模型...")
        self.model = tf.keras.models.load_model(model_path)
        logging.info("VOCA 模型成功載入!")

    @log_execution
    def predict_delta_pcds(self, subject_id, processed_audio):
        return self.model.predict(
            [
                np.repeat(subject_id, processed_audio.shape[0], axis=0),  # subject id = 0
                processed_audio,
            ]
        )
