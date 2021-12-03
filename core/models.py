# coding:utf-8

import tensorflow as tf

from tensorflow.python.keras.api._v2.keras import Input, Model
from tensorflow.python.keras.api._v2.keras.layers import Dense, \
                                           Conv2D, \
                                           Conv1D, \
                                           Conv3D, \
                                           Concatenate, \
                                           BatchNormalization, \
                                           LayerNormalization, \
                                           AveragePooling2D, \
                                           AveragePooling3D, \
                                           MaxPooling2D, \
                                           MaxPooling3D, \
                                           SeparableConv2D, \
                                           DepthwiseConv2D, \
                                           Activation, \
                                           SpatialDropout2D, \
                                           SpatialDropout3D, \
                                           Dropout, \
                                           AlphaDropout, \
                                           Flatten, \
                                           Lambda, \
                                           Multiply, \
                                           Reshape, \
                                           Add, \
                                           Subtract
from tensorflow.python.keras.api._v2.keras.constraints import max_norm, \
                                                              min_max_norm, \
                                                              unit_norm
from tensorflow.python.keras.api._v2.keras import backend as K

from core.regularizers import l_1, l_2, l1_l2, l2_1, tsc, sgl, tsgl, TSG

K.set_image_data_format('channels_last')


def get_custom_objects(name):
    if name in ('TSGLEEGNet'):
        return {'TSG': TSG}
    
    return {}


def get_compile(model: tf.keras.Model,
                optimizer=tf.keras.optimizers.Adam,
                lrate=1e-3,
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']):
    model.compile(optimizer=optimizer(lrate), loss=loss, metrics=metrics)
    
    return model


def TSGLEEGNet(nClasses,
               Chans,
               Samples,
               Colors,
               dropoutRate=0.5,
               kernLength=64,
               F1=9,
               D=4,
               F2=32,
               FSLength=16,
               l1=1e-4,
               l21=1e-4,
               tl1=1e-5,
               norm_rate=0.25,
               dtype=tf.float32,
               dropoutType='Dropout'):
    """
    An improvement of EEGNet. Using TSGL regularization.
    """
    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    elif dropoutType == 'AlphaDropout':
        dropoutType = AlphaDropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D, '
                         'AlphaDropout or Dropout, passed as a string.')
    # Learn from raw EEG signals
    _input_s = Input(shape=(Chans, Samples, Colors), dtype=dtype)
    s = Conv2D(F1, (1, kernLength),
               padding='same',
               use_bias=False,
               name='tfconv')(_input_s)
    s = BatchNormalization(axis=-1)(s)
    s = DepthwiseConv2D((Chans, 1),
                        use_bias=False,
                        depth_multiplier=D,
                        depthwise_constraint=max_norm(1.),
                        name='sconv')(s)
    s = BatchNormalization(axis=-1)(s)
    s = Activation('elu')(s)
    s = AveragePooling2D((1, 4))(s)
    s = dropoutType(dropoutRate)(s)
    s = Conv2D(F2, (1, FSLength),
               use_bias=False,
               padding='same',
               kernel_regularizer=sgl(l1, l21),
               activity_regularizer=tsc(tl1),
               name='fs')(s)
    s = BatchNormalization(axis=-1)(s)
    s = Activation('elu')(s)
    # s = AveragePooling2D((1, 8))(s)
    s = dropoutType(dropoutRate)(s)
    flatten = Flatten(name='flatten')(s)
    dense = Dense(
        nClasses,
        kernel_constraint=max_norm(norm_rate),
    )(flatten)
    _output_s = Activation('softmax', name='softmax')(dense)

    return Model(inputs=_input_s, outputs=_output_s, name='TSGLEEGNet')


def EEGNet(nb_classes,
           Chans=64,
           Samples=128,
           dropoutRate=0.5,
           kernLength=64,
           FSLength=16,
           F1=8,
           D=2,
           F2=16,
           norm_rate=0.25,
           dropoutType='Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/1dropout_rate088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:
        
        1. Depthwise Convolutions to learn spatial filters within a 
        temporal convolution. The use of the depth_multiplier option maps 
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn 
        spatial filters within each filter in a filter-bank. This also limits 
        the number of free parameters to fit when compared to a fully-connected
        convolution. 
        
        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions. 
        
    
    While the original paper used Dropout, we found that SpatialDropout2D 
    sometimes produced slightly better results for classification of ERP 
    signals. However, SpatialDropout2D significantly reduced performance 
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.
        
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 

    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == 'SpatialDropout2D':
        dropoutType = SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(F1, (1, kernLength),
                    padding='same',
                    use_bias=False,
                    name='tfconv')(input1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = DepthwiseConv2D((Chans, 1),
                             use_bias=False,
                             depth_multiplier=D,
                             depthwise_constraint=max_norm(1.),
                             name='sconv')(block1)
    block1 = BatchNormalization(axis=-1)(block1)
    block1 = Activation('elu')(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, FSLength),
                             padding='same',
                             use_bias=False,
                             name='fs')(block1)
    block2 = BatchNormalization(axis=-1)(block2)
    block2 = Activation('elu')(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name='flatten')(block2)

    dense = Dense(nb_classes,
                  name='dense',
                  kernel_constraint=max_norm(norm_rate))(flatten)
    softmax = Activation('softmax', name='softmax')(dense)

    return Model(inputs=input1, outputs=softmax, name='EEGNet')


def DeepConvNet(nb_classes, Chans=64, Samples=256, dropoutRate=0.5):
    """ Keras implementation of the Deep Convolutional Network as described in
    Schirrmeister et. al. (2017), Human Brain Mapping.
    
    This implementation assumes the input is a 2-second EEG signal sampled at 
    128Hz, as opposed to signals sampled at 250Hz as described in the original
    paper. We also perform temporal convolutions of length (1, 5) as opposed
    to (1, 10) due to this sampling rate difference. 
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                      ours        original paper
    pool_size        1, 2        1, 3
    strides          1, 2        1, 3
    conv filters     1, 5        1, 10
    
    Note that this implementation has not been verified by the original 
    authors. 
    
    """

    # start the model
    input_main = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(25, (1, 5),
                    input_shape=(1, Chans, Samples),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(25, (Chans, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation('elu')(block1)
    block1 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block1)
    block1 = Dropout(dropoutRate)(block1)

    block2 = Conv2D(50, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block2 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block2)
    block2 = Activation('elu')(block2)
    block2 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block2)
    block2 = Dropout(dropoutRate)(block2)

    block3 = Conv2D(100, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block2)
    block3 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block3)
    block3 = Activation('elu')(block3)
    block3 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block3)
    block3 = Dropout(dropoutRate)(block3)

    block4 = Conv2D(200, (1, 5),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block3)
    block4 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block4)
    block4 = Activation('elu')(block4)
    block4 = MaxPooling2D(pool_size=(1, 2), strides=(1, 2))(block4)
    block4 = Dropout(dropoutRate)(block4)

    flatten = Flatten()(block4)

    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax, name='DeepConvNet')


# need these for ShallowConvNet
def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(nb_classes, Chans=64, Samples=128, dropoutRate=0.5):
    """ Keras implementation of the Shallow Convolutional Network as described
    in Schirrmeister et. al. (2017), Human Brain Mapping.
    
    Assumes the input is a 2-second EEG signal sampled at 128Hz. Note that in 
    the original paper, they do temporal convolutions of length 25 for EEG
    data sampled at 250Hz. We instead use length 13 since the sampling rate is 
    roughly half of the 250Hz which the paper used. The pool_size and stride
    in later layers is also approximately half of what is used in the paper.
    
    Note that we use the max_norm constraint on all convolutional layers, as 
    well as the classification layer. We also change the defaults for the
    BatchNormalization layer. We used this based on a personal communication 
    with the original authors.
    
                     ours        original paper
    pool_size        1, 35       1, 75
    strides          1, 7        1, 15
    conv filters     1, 13       1, 25    
    
    Note that this implementation has not been verified by the original 
    authors. We do note that this implementation reproduces the results in the
    original paper with minor deviations. 
    """

    # start the model
    input_main = Input(shape=(Chans, Samples, 1))
    block1 = Conv2D(40, (1, 13),
                    input_shape=(Chans, Samples, 1),
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(input_main)
    block1 = Conv2D(40, (Chans, 1),
                    use_bias=False,
                    kernel_constraint=max_norm(2., axis=(0, 1, 2)))(block1)
    block1 = BatchNormalization(axis=-1, epsilon=1e-05, momentum=0.1)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 35), strides=(1, 7))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(dropoutRate)(block1)
    flatten = Flatten()(block1)
    dense = Dense(nb_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation('softmax')(dense)

    return Model(inputs=input_main, outputs=softmax, name='ShallowConvNet')


# Multi-branch 3D CNN from `A Multi-Branch 3D Convolutional Neural Network
# for EEG-Based Motor Imagery Classification`, IEEE TRANSACTIONS ON NEURAL
# SYSTEMS AND REHABILITATION ENGINEERING, VOL. 27, NO. 10, OCTOBER 2019
def SRF3DCNN(nClasses, H, W, Samples):
    _input = Input(shape=(Samples, H, W, 16), name='SRF_Input')
    srf = Conv3D(32, (1, 2, 2), strides=(1, 2, 2), padding='same')(_input)
    srf = BatchNormalization(axis=-1)(srf)
    srf = Activation('elu')(srf)
    # srf = SpatialDropout3D(0.5)(srf)
    srf = Conv3D(64, (1, 2, 2), strides=(1, 2, 2), padding='same')(srf)
    srf = BatchNormalization(axis=-1)(srf)
    srf = Activation('elu')(srf)
    # srf = SpatialDropout3D(0.5)(srf)
    flatten = Flatten()(srf)
    dense = Dense(32)(flatten)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(32)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(nClasses)(dense)
    _output = Activation('softmax', name='SRF_Output')(dense)
    return Model(inputs=_input, outputs=_output, name='SRF')


def MRF3DCNN(nClasses, H, W, Samples):
    _input = Input(shape=(Samples, H, W, 16), name='MRF_Input')
    mrf = Conv3D(32, (3, 2, 2), strides=(2, 2, 2), padding='same')(_input)
    mrf = BatchNormalization(axis=-1)(mrf)
    mrf = Activation('elu')(mrf)
    # mrf = SpatialDropout3D(0.5)(mrf)
    mrf = Conv3D(64, (3, 2, 2), strides=(2, 2, 2), padding='same')(mrf)
    mrf = BatchNormalization(axis=-1)(mrf)
    mrf = Activation('elu')(mrf)
    # mrf = SpatialDropout3D(0.5)(mrf)
    flatten = Flatten()(mrf)
    dense = Dense(32)(flatten)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(32)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(nClasses)(dense)
    _output = Activation('softmax', name='MRF_Output')(dense)
    return Model(inputs=_input, outputs=_output, name='MRF')


def LRF3DCNN(nClasses, H, W, Samples):
    _input = Input(shape=(Samples, H, W, 16), name='LRF_Input')
    lrf = Conv3D(32, (5, 2, 2), strides=(4, 2, 2), padding='same')(_input)
    lrf = BatchNormalization(axis=-1)(lrf)
    lrf = Activation('elu')(lrf)
    # lrf = SpatialDropout3D(0.5)(lrf)
    lrf = Conv3D(64, (5, 2, 2), strides=(4, 2, 2), padding='same')(lrf)
    lrf = BatchNormalization(axis=-1)(lrf)
    lrf = Activation('elu')(lrf)
    # lrf = SpatialDropout3D(0.5)(lrf)
    flatten = Flatten()(lrf)
    dense = Dense(32)(flatten)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(32)(dense)
    dense = BatchNormalization()(dense)
    dense = Activation('relu')(dense)
    dense = Dense(nClasses)(dense)
    _output = Activation('softmax', name='LRF_Output')(dense)
    return Model(inputs=_input, outputs=_output, name='LRF')


def MB3DCNN(nClasses, H, W, Samples):
    _input = Input(shape=(Samples, H, W, 1), name='MB_Input')
    mb = Conv3D(16, (5, 3, 3), strides=(4, 2, 2), padding='same')(_input)
    mb = BatchNormalization(axis=-1)(mb)
    mb = Activation('elu')(mb)
    # mb = SpatialDropout3D(0.5)(mb)

    _srf_output = SRF3DCNN(nClasses, mb.shape[2], mb.shape[3], mb.shape[1])(mb)
    _mrf_output = MRF3DCNN(nClasses, mb.shape[2], mb.shape[3], mb.shape[1])(mb)
    _lrf_output = LRF3DCNN(nClasses, mb.shape[2], mb.shape[3], mb.shape[1])(mb)

    _add = Add()([_srf_output, _mrf_output, _lrf_output])
    # _output = Activation('softmax', name='MB_Output')(_add)
    _output = Dense(nClasses, activation='softmax', name='MB_Output')(_add)
    return Model(inputs=_input, outputs=[_output], name='MB3DCNN')
