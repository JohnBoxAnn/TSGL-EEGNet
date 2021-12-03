# coding:utf-8

import tensorflow as tf

from tensorflow.python.keras.api._v2.keras.initializers import Initializer


class EmbeddingInit(Initializer):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __call__(self, shape, dtype):
        if not self.embeddings.shape == shape:
            raise ValueError(self.embeddings.shape, shape)
        return tf.convert_to_tensor(self.embeddings, dtype=dtype)

    def get_config(self):
        config = {'embeddings': self.embeddings}
        base_config = super().get_config()
        base_config.update(config)
        return base_config