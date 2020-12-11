# coding:utf-8
import tensorflow as tf
from tensorflow.python.keras.api._v2.keras.layers import Softmax
from tensorflow.python.keras.api._v2.keras.layers import Dense
from tensorflow.python.keras.api._v2.keras.layers import Attention
from tensorflow.python.keras.api._v2.keras.layers import Multiply
from tensorflow.python.keras.api._v2.keras.layers import Layer
from tensorflow.python.keras.api._v2.keras import Model
from tensorflow.python.keras.api._v2.keras import initializers, regularizers, constraints
from tensorflow.python.keras.api._v2.keras import backend as K

from core.regularizers import TSG as _TSG


# TODO: Construct attention layers
class BaseAttention(Layer):
    '''Dense Attention'''
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='BaseAttention',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'), )

        super().__init__(
            trainable=True,
            name=name,
            activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)

        self.axis = axis
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def build(self, input_shape):
        self.w = self.add_weight(name='kernel',
                                 shape=(input_shape[self.axis],
                                        input_shape[self.axis]),
                                 dtype=self.dtype,
                                 initializer=self.kernel_initializer,
                                 regularizer=self.kernel_regularizer,
                                 constraint=self.kernel_constraint,
                                 trainable=True)
        if self.use_bias:
            self.b = self.add_weight(name='bias',
                                     shape=(input_shape[self.axis], ),
                                     dtype=self.dtype,
                                     initializer=self.bias_initializer,
                                     regularizer=self.bias_regularizer,
                                     constraint=self.bias_constraint,
                                     trainable=True)
        self.c = self.add_weight(name='constant',
                                 shape=(input_shape[-1], input_shape[-1]),
                                 dtype=self.dtype,
                                 initializer=initializers.get('identity'),
                                 trainable=False)
        super().build(input_shape)

    def call(self, inputs):
        print(inputs.shape)
        t = tf.tensordot(inputs, self.w, [[self.axis], [0]])
        print(t.shape)
        if self.use_bias:
            K.bias_add(t, self.b)
        softmax = K.softmax(t, axis=-1)
        print(softmax.shape)
        softmax = tf.tensordot(softmax, self.c, [[self.axis], [0]])
        print(softmax.shape)
        inputs = tf.multiply(inputs, softmax)
        print(inputs.shape)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'axis': self.axis,
            'supports_masking': self.supports_masking,
            'use_bias': self.use_bias,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer,
            'bias_regularizer': self.bias_regularizer,
            'activity_regularizer': self.activity_regularizer,
            'kernel_constraint': self.kernel_constraint,
            'bias_constraint': self.bias_constraint
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rawEEGAttention(BaseAttention):
    '''Attention for raw EEG data'''
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='rawEEGAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class rawEEGTimeAttention(BaseAttention):
    #TODO
    def __init__(self,
                 steps=25,
                 winLength=1000,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='rawEEGTimeAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)
        self.steps = steps
        self.winLength = winLength

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        t = tf.tensordot(inputs, self.w, [[self.axis], [0]])
        if self.use_bias:
            K.bias_add(t, self.b)
        softmax = K.softmax(t, axis=-1)
        softmax = tf.tensordot(softmax, self.c, [[self.axis], [0]])
        inputs = tf.multiply(inputs, softmax)
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class graphEEGAttention(BaseAttention):
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='graphEEGAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class BaseSelfAttention(BaseAttention):
    '''Dense Self Attention'''
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='BaseSelfAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        att = []
        for i in range(inputs.shape[self.axis]):
            relation = self._relationFunc(inputs, axis=self.axis, num=i)
            softmax = K.softmax(relation)
            att.append(tf.reduce_sum(tf.multiply(relation, softmax)).numpy())
        return K.constant(att, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _relationFunc(self, axis, num):
        raise NotImplementedError


class rawEEGSelfAttention(BaseSelfAttention):
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='rawEEGSelfAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _relationFunc(self, axis, num):
        pass


class graphEEGSelfAttention(BaseSelfAttention):
    def __init__(self,
                 axis=-1,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name='graphEEGSelfAttention',
                 **kwargs):
        super().__init__(axis=axis,
                         use_bias=use_bias,
                         kernel_initializer=kernel_initializer,
                         bias_initializer=bias_initializer,
                         kernel_regularizer=kernel_regularizer,
                         bias_regularizer=bias_regularizer,
                         activity_regularizer=activity_regularizer,
                         kernel_constraint=kernel_constraint,
                         bias_constraint=bias_constraint,
                         name=name,
                         **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        return super().call(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def _relationFunc(self, axis, num):
        pass


class TSGRegularization(Layer):
    """
    Layer that applies an update to the cost function based input activity.

    Arguments:
        l1  : float, Positive L1 regularization factor.
        l21 : float, Positive L21 regularization factor.
        tl1 : float, Positive TL1 regularization factor.

    Input shape:
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    Output shape:
        Same shape as input.
  """
    def __init__(self, l1=0., l21=0., tl1=0., **kwargs):
        super().__init__(activity_regularizer=_TSG(l1=l1, l21=l21, tl1=tl1),
                         **kwargs)
        self.supports_masking = True
        self.l1 = l1
        self.l21 = l21
        self.tl1 = tl1

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'l1': self.l1, 'l21': self.l21, 'tl1': self.tl1}
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))