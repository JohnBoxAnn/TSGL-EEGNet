#coding:utf-8

import os
import timeit
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from core.train import create_EEGNet, create_TSGLEEGNet, crossValidate, gridSearch
from core.generators import rawGenerator, graphGenerator
from core.splits import StratifiedKFold, AllTrain

from tensorflow.python.keras.api._v2.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
K.set_image_data_format('channels_last')
srate = 250


def time_format(secs):
    mins = int(secs // 60)
    secs %= 60
    hours = mins // 60
    mins %= 60
    days = hours // 24
    hours %= 24
    return days, hours, mins, secs


start = timeit.default_timer()
crossValidate(create_TSGLEEGNet,
              rawGenerator,
              StratifiedKFold,
              kFold=3,
              shuffle=True,
              normalizing=True,
              preserve_initfile=False,
              reinit=True,
              cropping=True,
              winLength=2 * srate,
              step=1,
              beg=0,
              end=4,
              srate=srate,
              epochs=200,
              patience=50)(4, F=16, D=10, Ns=20, FSLength=1)

# parameters = {
#     'l1': {
#         '1': [2.5e-5],
#         '2': [1e-3],
#         '3': [1e-4],
#         '4': [7.5e-5],
#         '5': [2.5e-5],
#         '6': [5e-5],
#         '7': [7.5e-5],
#         '8': [1e-3],
#         '9': [7.5e-5]
#     },
#     'l21':
#     {
#         '1': [2.5e-5],
#         '2': [1e-4],
#         '3': [7.5e-5],
#         '4': [1e-4],
#         '5': [1e-4],
#         '6': [1e-4],
#         '7': [1e-4],
#         '8': [1e-4],
#         '9': [1e-4]
#     },
#     'tl1': {
#         '1': [7.5e-6],
#         '2': [7.5e-6],
#         '3': [2.5e-6],
#         '4': [1e-5],
#         '5': [7.5e-6],
#         '6': [1e-6],
#         '7': [2.5e-6],
#         '8': [5e-6],
#         '9': [2.5e-5]
#     }
# }
# gridSearch(create_rawEEGConvNet,
#            parameters,
#            rawGenerator,
#            StratifiedKFold,
#            kFold=10,
#            subs=range(1, 10),
#            shuffle=True,
#            normalizing=True,
#            preserve_initfile=False,
#            reinit=True,
#            cropping=True,
#            beg=0,
#            end=4,
#            srate=srate,
#            epochs=10, # change them
#            patience=3)(4, F=16, D=10, Ns=20)

end = timeit.default_timer()
print("Time used: {0:0>2d}d {1:0>2d}h {2:0>2d}m {3:.4f}s".format(
    *time_format(end - start)))