# coding:utf-8
import os
import timeit
import tensorflow as tf
from tensorflow.python.keras.api._v2.keras import backend as K

from core.get_model import create_EEGNet, create_TSGLEEGNet
from core.training import crossValidate, gridSearch
from core.dataloaders import RawDataloader
from core.generators import RawGenerator
from core.splits import StratifiedKFold, AllTrain
from core.metrics import Kappa

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
K.set_image_data_format('channels_last')
srate = 250
num_classes = 4
batch_size = 10


def time_format(secs):
    mins = int(secs // 60)
    secs %= 60
    hours = mins // 60
    mins %= 60
    days = hours // 24
    hours %= 24
    return days, hours, mins, secs


train_datapath = os.path.join('data', 'A', 'TrainSet', 'example_data.mat')
test_datapath = os.path.join('data', 'A', 'TestSet', 'example_data.mat')
datadir = None
# train_datapath = None
# test_datapath = None
# datadir = os.path.join('data', 'A')

start = timeit.default_timer()
# Change kFold, epochs and patience to get higher acc
crossValidate(
    create_TSGLEEGNet,
    dataLoader=RawDataloader,
    splitMethod=AllTrain,
    dataGent=RawGenerator,
    traindata_filepath=train_datapath,
    testdata_filepath=test_datapath,
    datadir=datadir,
    kFold=5,
    # If use 'traindata_filepath' or 'testdata_filepath', set subs=[1]
    subs=[1],
    shuffle=True,
    norm_mode='z-score',
    preserve_initfile=False,
    reinit=True,
    # If needed, turn cropping on.
    # But its accuracy evaluation indicator is not clear.
    cropping=False,
    cpt=0.5,
    step=int(0.2 * srate),
    max_crop=6,
    beg=0.,
    end=4.,
    srate=srate,
    batch_size=batch_size,
    epochs=1200,
    patience=300)(
        nClasses=num_classes,
        Chans=22,
        F=16,
        D=10,
        Ns=20,
        l1=1e-4,
        l21=7.5e-5,
        tl1=2.5e-6,
        metrics=[
            'accuracy',
            Kappa(num_classes, sparse_labels=True)
        ],
        lrate=1e-3,
    )

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
# OR
# parameters = {
#     'l1': {
#         # '1': [5e-3],
#         '2':
#         list(np.linspace(1e-2, 2.5e-3, 4)) +
#         list(np.linspace(1e-3, 2.5e-4, 4)) +
#         list(np.linspace(1e-4, 2.5e-5, 4)) + [1e-5, 0.],
#         # '3': [7.5e-4]
#     },
#     'l21': [1e-3],
#     'tl1': {
#         # '1': [7.5e-4],
#         '2': [2.5e-5],
#         # '3': [7.5e-4]
#     }
# }
# # OR mix them
# gridSearch(
#     create_TSGLEEGNet,
#     parameters,
#     dataLoader=RawDataloader,
#     splitMethod=AllTrain,
#     dataGent=RawGenerator,
#     traindata_filepath=train_datapath,
#     testdata_filepath=test_datapath,
#     datadir=datadir,
#     kFold=5,
#     subs=range(2, 3),
#     shuffle=True,
#     norm_mode='z-score',
#     preserve_initfile=False,
#     reinit=True,
#     cpt=0.5,
#     step=int(0.2 * srate),
#     max_crop=6,
#     beg=0.,
#     end=4.,
#     srate=srate,
#     epochs=1200,  # change them
#     patience=300)(4, Chans=60, F=16, D=10, Ns=20)

end = timeit.default_timer()
print("Time used: {0:0>2d}d {1:0>2d}h {2:0>2d}m {3:.4f}s".format(
    *time_format(end - start)))