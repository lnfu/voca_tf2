import numpy as np
import tensorflow as tf

import os
import pickle
import logging
import resampy
from python_speech_features import mfcc


class AudioHandler:
    def __init__(self):
        self.num_features = 29
        self.window_size = 16
        self.stride = 1

        self.processed_data_path = "data/deepspeech.pkl"
        self.raw_data_path = "data/raw_audio_fixed.pkl"

    def resample(self, input, sample_rate_in, sample_rate_out, output_length):
        input_length = input.shape[0]
        num_features = input.shape[1]

        duration = input_length / float(sample_rate_in)

        if not output_length:
            output_length = int(duration * sample_rate_out)

        input_timestamps = np.arange(input_length) / float(sample_rate_in)
        output_timestamps = np.arange(output_length) / float(sample_rate_out)

        output_features = np.zeros((output_length, num_features))
        for feat in range(num_features):
            output_features[:, feat] = np.interp(
                output_timestamps, input_timestamps, input[:, feat]
            )
        return output_features

    def get_processed_training_data(self):
        if os.path.exists(self.processed_data_path):
            logging.info(f"使用已處理過的音訊 {self.processed_data_path}")
            return pickle.load(open(self.processed_data_path, "rb"), encoding="latin1")

        logging.info(f"音訊尚未處理")
        # 也許就直接搞自己的 dataset 就不用多這些步驟?
        # 換個角度想, 不要動　dataset 好像比較合理?
        # TODO 檢查　raw_data_path 使否存在
        raw_data = pickle.load(open(self.raw_data_path, "rb"), encoding="latin1")

        processed_data = self.batch_process(raw_data)
        pickle.dump(
            processed_data, open(self.processed_data_path, "wb")
        )  # save processed_data

        return processed_data

    # TODO refactor (封裝)
    def batch_process(self, raw_data):

        logging.info(f"開始處理音訊")

        # 先載入 deepspeech 模型 (好像是 6 層)
        # 輸入是 MFCC
        # 輸出是 ??

        graph = tf.Graph()

        logging.info("正在載入 DeepSpeech 模型...")
        with graph.as_default():
            with tf.io.gfile.GFile("data/output_graph.pb", "rb") as f:
                graph_def = tf.compat.v1.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.compat.v1.import_graph_def(graph_def, name="deepspeech")

        input_tensor = graph.get_tensor_by_name("deepspeech/input_node:0")
        input_length = graph.get_tensor_by_name("deepspeech/input_lengths:0")
        logits = graph.get_tensor_by_name("deepspeech/logits:0")  # output
        logging.info("DeepSpeech 模型成功載入!")

        processed_data = {}

        with tf.compat.v1.Session(graph=graph) as sess:

            for subject_name, subject_data in raw_data.items():
                processed_subject_data = {}

                for sequence_name, sequence_data in subject_data.items():

                    sample_rate = sequence_data["sample_rate"]  # 22000 = 22 kHz
                    raw_audio = sequence_data[
                        "audio"
                    ]  # ndarray, shape=(?,), dtype=int16
                    logging.info(
                        f"當前處理音訊: Subject = {subject_name}, Sequence = {sequence_name}"
                    )

                    resampled_audio = resampy.resample(
                        raw_audio.astype(np.float32),
                        sample_rate,
                        16000,  # TODO why 16000?
                    )

                    # mfcc_features shape = (x, 26)
                    mfcc_features = mfcc(
                        resampled_audio.astype(np.int16),
                        samplerate=16000,
                        numcep=26,  # 26 = num of MFCC audio features
                    )

                    # We only keep every second feature (BiRNN stride = 2)
                    # 所以只要每兩個取一個就好
                    # shape = (x/2, 26)
                    mfcc_features = mfcc_features[::2]

                    num = len(mfcc_features)  # TODO 看懂後改名

                    # 頭尾加上空白
                    # shape = (9 + x/2 + 9, 26)
                    zero_pad = np.zeros((9, 26), dtype=mfcc_features.dtype)
                    mfcc_features = np.concatenate((zero_pad, mfcc_features, zero_pad))

                    # TODO 看不懂這邊, 之後有時間再研究
                    # TODO 可能 deepspeech_input 改個名稱
                    deepspeech_input = np.lib.stride_tricks.as_strided(
                        mfcc_features,
                        (num, 2 * 9 + 1, 26),  # ?, 19, 26
                        (
                            mfcc_features.strides[0],
                            mfcc_features.strides[0],
                            mfcc_features.strides[1],
                        ),
                        writeable=False,
                    )
                    deepspeech_input = np.reshape(deepspeech_input, [num, -1])
                    deepspeech_input = np.copy(
                        deepspeech_input
                    )  # 因為前面用 np.lib.stride_tricks.as_strided 只是回傳 view
                    deepspeech_input = (
                        deepspeech_input - np.mean(deepspeech_input)
                    ) / np.std(deepspeech_input)

                    deepspeech_output = sess.run(
                        logits,
                        feed_dict={
                            input_tensor: deepspeech_input[np.newaxis, ...],
                            input_length: [deepspeech_input.shape[0]],
                        },
                    )  # shape = (?, 29)

                    # Resample network output from 50 fps to 60 fps
                    duration = float(raw_audio.shape[0]) / sample_rate
                    # audio_len_s = float(audio_sample.shape[0]) / sample_rate
                    num_frames = int(round(duration * 60))
                    deepspeech_output = self.resample(
                        deepspeech_output[:, 0], 50, 60, output_length=num_frames
                    )

                    self.window_size = 16  # TODO
                    self.stride = 1  # TODO

                    zero_pad = np.zeros(
                        (int(self.window_size // 2), deepspeech_output.shape[1])
                    )
                    deepspeech_output = np.concatenate(
                        (zero_pad, deepspeech_output, zero_pad), axis=0
                    )

                    processed_sequence_data = []
                    for idx in range(
                        0, deepspeech_output.shape[0] - self.window_size, self.stride
                    ):
                        processed_sequence_data.append(
                            deepspeech_output[idx : idx + self.window_size]
                        )

                    processed_subject_data[sequence_name] = np.array(
                        processed_sequence_data
                    )

                processed_data[subject_name] = processed_subject_data

        logging.info("音訊處理完成")
        return processed_data
