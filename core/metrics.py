# coding:utf-8
'''`tensorflow-addons.metrics.CohenKappa` is rewritten to fix some compatibility problems'''
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow_addons.metrics import CohenKappa


class Kappa(CohenKappa):
    '''Compute kappa value as a metric'''
    def __new__(cls, *args, **kwargs):
        ck = super().__new__(cls, *args, **kwargs)
        if tf.version.VERSION >= '2.4.0':
            if ck.reset_states:
                ck.reset_state = ck.reset_states
                ck.reset_states = None
        return ck

    def result(self):
        # Here raises Error.
        # nb_ratings = tf.shape(self.conf_mtx)[0]
        # When using symbolic tensors, such as when using the Keras API,
        # tf.shape() will return the shape of the symbolic tensor.
        # In these cases, using tf.Tensor.shape will return more informative
        # results.
        nb_ratings = self.conf_mtx.shape[0]
        weight_mtx = tf.ones([nb_ratings, nb_ratings], dtype=tf.float32)

        # 2. Create a weight matrix
        if self.weightage is None:
            diagonal = tf.zeros([nb_ratings], dtype=tf.float32)
            weight_mtx = tf.linalg.set_diag(weight_mtx, diagonal=diagonal)
        else:
            weight_mtx += tf.cast(tf.range(nb_ratings), dtype=tf.float32)
            weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

            if self.weightage == "linear":
                weight_mtx = tf.abs(weight_mtx - tf.transpose(weight_mtx))
            else:
                weight_mtx = tf.pow((weight_mtx - tf.transpose(weight_mtx)), 2)

        weight_mtx = tf.cast(weight_mtx, dtype=self.dtype)

        # 3. Get counts
        actual_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=1)
        pred_ratings_hist = tf.reduce_sum(self.conf_mtx, axis=0)

        # 4. Get the outer product
        out_prod = pred_ratings_hist[..., None] * actual_ratings_hist[None,
                                                                      ...]

        # 5. Normalize the confusion matrix and outer product
        conf_mtx = self.conf_mtx / tf.reduce_sum(self.conf_mtx)
        out_prod = out_prod / tf.reduce_sum(out_prod)

        conf_mtx = tf.cast(conf_mtx, dtype=self.dtype)
        out_prod = tf.cast(out_prod, dtype=self.dtype)

        # 6. Calculate Kappa score
        numerator = tf.reduce_sum(conf_mtx * weight_mtx)
        denominator = tf.reduce_sum(out_prod * weight_mtx)
        return tf.cond(
            tf.math.is_nan(denominator),
            true_fn=lambda: 0.0,
            false_fn=lambda: 1 - (numerator / denominator),
        )

    def reset_states(self):
        """Resets all of the metric state variables."""
        K.batch_set_value([(v, np.zeros((self.num_classes, self.num_classes)))
                           for v in self.variables])
