import numpy as np
import chumpy as ch
import tensorflow as tf

import pickle
import logging

model_path = "./data/generic_model.pkl"


class Flame:
    @classmethod
    def load(clazz):
        model = pickle.load(open(model_path, "rb"), encoding="latin1")

        clazz.v_template = tf.convert_to_tensor(model["v_template"]) # (5023, 3)
        clazz.shapedirs = tf.convert_to_tensor(model["shapedirs"]) # (5023, 3, 400)
        print(type(clazz.shapedirs))
        print(clazz.shapedirs.shape)
        exit(0)

        # J_regressor_coo = model["J_regressor"].tocoo()
        # clazz.J_regressor = tf.SparseTensor(
        #     indices=np.vstack((J_regressor_coo.row, J_regressor_coo.col)).T,
        #     values=J_regressor_coo.data,
        #     dense_shape=J_regressor_coo.shape,
        # )

        logging.info(f"loaded FLAME model from: {model_path}")


Flame.load()
