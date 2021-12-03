# coding:utf-8
import tensorflow as tf

from core.models import EEGNet, TSGLEEGNet, MB3DCNN
from core.models import get_compile


def create_MB3DCNN(nClasses,
                   H,
                   W,
                   Samples,
                   optimizer=tf.keras.optimizers.Adam,
                   lrate=1e-3,
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'],
                   summary=True):
    model = MB3DCNN(nClasses, H=H, W=W, Samples=Samples)
    model = get_compile(model,
                        optimizer=optimizer,
                        lrate=lrate,
                        loss=loss,
                        metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, model.name + '.png', show_shapes=True)
    return model


def create_EEGNet(nClasses,
                  Samples,
                  Chans=22,
                  F=9,
                  D=4,
                  Ns=4,
                  kernLength=64,
                  FSLength=16,
                  dropoutRate=0.5,
                  optimizer=tf.keras.optimizers.Adam,
                  lrate=1e-3,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  summary=True):
    model = EEGNet(nClasses,
                   Chans=Chans,
                   Samples=Samples,
                   kernLength=kernLength,
                   FSLength=FSLength,
                   dropoutRate=dropoutRate,
                   F1=F,
                   D=D,
                   F2=nClasses * 2 * Ns)
    model = get_compile(model,
                        optimizer=optimizer,
                        lrate=lrate,
                        loss=loss,
                        metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, model.name + '.png', show_shapes=True)
    return model


def create_TSGLEEGNet(nClasses,
                      Samples,
                      Chans=22,
                      Colors=1,
                      F=9,
                      D=4,
                      Ns=4,
                      kernLength=64,
                      FSLength=16,
                      dropoutRate=0.5,
                      l1=1e-4,
                      l21=1e-4,
                      tl1=1e-5,
                      optimizer=tf.keras.optimizers.Adam,
                      lrate=1e-3,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'],
                      summary=True):
    model = TSGLEEGNet(nClasses,
                       Chans=Chans,
                       Samples=Samples,
                       Colors=Colors,
                       kernLength=kernLength,
                       FSLength=FSLength,
                       dropoutRate=dropoutRate,
                       F1=F,
                       D=D,
                       F2=nClasses * 2 * Ns,
                       l1=l1,
                       l21=l21,
                       tl1=tl1)
    model = get_compile(model,
                        optimizer=optimizer,
                        lrate=lrate,
                        loss=loss,
                        metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, model.name + '.png', show_shapes=True)
    return model