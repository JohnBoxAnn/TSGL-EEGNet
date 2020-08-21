# coding:utf-8
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.api._v2.keras.regularizers import Regularizer
from tensorflow.python.keras.api._v2.keras.regularizers import l1 as l_1
from tensorflow.python.keras.api._v2.keras.regularizers import l2 as l_2
from tensorflow.python.keras.api._v2.keras.regularizers import l1_l2
from tensorflow.python.keras.api._v2.keras import backend as K


class TSG(Regularizer):
    '''
    Regularizer for TSG regularization.

    Parameters
    ----------
    ```txt
    l1              : float, Positive L1 regularization factor.
    l21             : float, Positive L21 regularization factor.
    tl1             : float, Positive TL1 regularization factor.
    ```

    Return
    ------
    ```txt
    regularization  : float, Regularization fine.
    ```
    '''
    def __init__(self, l1=0., l21=0., tl1=0.):  # pylint: disable=redefined-outer-name
        self.l1 = K.cast_to_floatx(l1)
        self.l21 = K.cast_to_floatx(l21)
        self.tl1 = K.cast_to_floatx(tl1)

    def __call__(self, x):
        if not self.l1 and not self.l21 and not self.tl1:
            return K.constant(0.)
        regularization = 0.

        # TODO: should we seperate kernel and activaty regularizers?
        if x.shape[0] == 1:  # shape (1, len, Inputs, Outputs)
            ntf = tf.squeeze(x, 0)  # shape (len, Inputs, Outputs)
        elif x.shape[1] == 1:  # shape (?, 1, Timesteps, Features)
            ntf = tf.squeeze(x, 1)  # shape (?, Timesteps, Features)
        elif len(x.shape) == 2:  # shape (Inputs, Outputs)
            ntf = tf.expand_dims(x, axis=0)  # shape (1, Inputs, Outputs)
        else:  # shape (?, Inputs, Outputs)
            ntf = x  # shape (?, Inputs, Outputs)
        # now Tensor `ntf` ranks 3

        if self.l1:
            regularization += self.l1 * tf.reduce_sum(tf.abs(ntf))
        if self.l21:
            regularization += self.l21 * tf.reduce_sum(
                tf.sqrt(
                    tf.multiply(
                        tf.cast(ntf.shape[0] * ntf.shape[1], tf.float32),
                        tf.reduce_sum(tf.square(ntf), [0, 1]))))
        if self.tl1:
            regularization += self.tl1 * tf.reduce_sum(
                tf.abs(tf.subtract(ntf[:, :-1, :], ntf[:, 1:, :])))
        return regularization

    def get_config(self):
        return {
            'l1': float(self.l1),
            'l21': float(self.l21),
            'tl1': float(self.tl1)
        }


# l2_1 = group lasso
def l2_1(l21=0.01):
    '''group lasso'''
    return TSG(l21=l21)


# to preserve the temporal smoothness
def tsc(tl1=0.01):
    '''
    temporal constrained to preserve the temporal smoothness, for 
    activity_regularizer.
    '''
    return TSG(tl1=tl1)


# l1 + l2_1 = sparse group lasso
def sgl(l1=0.01, l21=0.01):
    '''sparse group lasso, for kernel_regularizer'''
    return TSG(l1=l1, l21=l21)


def tsgl(l1=0.01, l21=0.01, tl1=0.01):
    '''
    temporal constrained sparse group lasso, use tsc + sgl instead.
    '''
    return TSG(l1=l1, l21=l21, tl1=tl1)
