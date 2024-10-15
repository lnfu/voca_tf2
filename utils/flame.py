import numpy as np
import chumpy as ch
import tensorflow as tf

import pickle
import logging

model_path = "./data/generic_model.pkl"


def rotation_vector_to_matrix(vec):
    theta = tf.norm(vec)
    n = vec / theta

    R1 = tf.cos(theta) * tf.eye(3)
    R2 = tf.einsum("i,j->ij", n, n) * (1 - tf.cos(theta))
    R3 = tf.convert_to_tensor([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]], dtype=tf.float32) * tf.sin(theta)

    return R1 + R2 + R3


class Flame:
    @classmethod
    def load(clazz):
        model = pickle.load(open(model_path, "rb"), encoding="latin1")

        clazz.v_template = tf.convert_to_tensor(model["v_template"], dtype=tf.float32)  # (5023, 3)
        clazz.shapedirs = tf.convert_to_tensor(model["shapedirs"], dtype=tf.float32)  # (5023, 3, 400)

        J_regressor_coo = model["J_regressor"].tocoo()
        clazz.J_regressor = tf.SparseTensor(
            indices=np.vstack((J_regressor_coo.row, J_regressor_coo.col)).T,
            values=tf.convert_to_tensor(J_regressor_coo.data, dtype=tf.float32),
            dense_shape=J_regressor_coo.shape,
        )  # (5, 5023)
        clazz.posedirs = tf.convert_to_tensor(model["posedirs"], dtype=tf.float32)  # (5023, 3, 36)

        kintree_table = model["kintree_table"]
        clazz.parent = {}
        for i in range(len(kintree_table[0])):
            if kintree_table[0][i] == 0xFFFFFFFF:  # root
                continue
            clazz.parent[kintree_table[1][i]] = kintree_table[0][i]

        clazz.weights = tf.convert_to_tensor(model["weights"], dtype=tf.float32)

        logging.info(f"loaded FLAME model from: {model_path}")

    @classmethod
    def calculate_pcd_by_param(clazz, flame_params: tf.Tensor) -> tf.Tensor:

        betas = flame_params[:400]  # shape (300) + expression (100)
        pose = flame_params[400:415]

        # pose = (15,)
        # betas = (400,) -> shape + expression
        v_shaped = tf.reduce_sum(clazz.shapedirs * betas) + clazz.v_template  # (5023, 3)

        J = tf.sparse.sparse_dense_matmul(clazz.J_regressor, v_shaped)  # (5, 3)

        pose_1_to_4 = tf.reshape(pose[1 * 3 :], (4, 3))  # 拿掉第一個 (global rotation)
        lrotmin = tf.reshape(tf.map_fn(lambda x: rotation_vector_to_matrix(x), pose_1_to_4), (4 * 3 * 3,))  # (36,)

        v_posed = v_shaped + tf.reduce_sum(clazz.posedirs * lrotmin)  # (5023, 3)
        homo_v_posed = tf.concat([v_posed, tf.ones((5023, 1))], axis=1)  # (5023, 4)

        poses = [pose[i * 3 : (i + 1) * 3] for i in range(len(pose) // 3)]

        A = [None] * 5
        A[0] = tf.concat(
            [
                tf.concat([rotation_vector_to_matrix(poses[0]), tf.reshape(J[0], (3, 1))], axis=1),
                tf.constant([[0.0, 0.0, 0.0, 1.0]]),
            ],
            axis=0,
        )
        for i in range(1, 5):
            j = clazz.parent[i]
            A[i] = tf.matmul(
                A[j],
                tf.concat(
                    [
                        tf.concat([rotation_vector_to_matrix(poses[i]), tf.reshape(J[i] - J[j], (3, 1))], axis=1),
                        tf.constant([[0.0, 0.0, 0.0, 1.0]]),
                    ],
                    axis=0,
                ),
                transpose_b=True,
            )
        A = tf.stack(A, axis=-1)
        T = tf.matmul(A, clazz.weights, transpose_b=True)  # (4, 4, 5023)

        results = tf.matmul(
            tf.transpose(T, (2, 0, 1)), tf.expand_dims(homo_v_posed, axis=2)
        )  # (4, 4, 5023) * (5023, 4, 1) = (5023, 4, 1)
        results = tf.squeeze(results, axis=-1)  # (5023, 4,)

        assert results.shape == (
            5023,
            4,
        )

        return results[:, :3]  # (5023, 3,)


Flame.load()
