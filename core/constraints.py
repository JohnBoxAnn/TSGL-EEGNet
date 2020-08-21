# coding:utf-8
"""
TODO: kernel constraint to keep mean of kernel weights values 0, 
      and variance of them values 1/K (where K means the number 
      of input's neurons).
"""
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.api._v2.keras.constraints import Constraint
from tensorflow.python.keras.api._v2.keras import backend as K


class StdNorm(Constraint):
    def __init__(self, axis=None):
        self.axis = axis

    # @tf.function
    def __call__(self, w):
        size = tf.shape(w, out_type=tf.float32)
        k = tf.constant(1., tf.float32)
        for i in range(size.shape[0] - 1):
            k = tf.multiply(k, size[i])
        mu = tf.math.reduce_mean(w, axis=self.axis, keepdims=True)
        std = tf.math.reduce_std(w, axis=self.axis, keepdims=True)
        mu = tf.multiply(tf.ones_like(w), mu)
        std = tf.multiply(tf.ones_like(w), std)
        return tf.divide(tf.subtract(w, mu), tf.multiply(std, tf.sqrt(k)))

    def get_config(self):
        return {'axis': self.axis}


std_norm = StdNorm