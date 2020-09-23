# coding:utf-8
import os
import numpy as np

from core.utils import load_data
from core.utils import computeKappa
from core.visualization import visualize
from core.regularizers import TSG
from core.train import crossValidate, create_EEGNet

from tensorflow.python.keras.api._v2.keras.models import load_model
from tensorflow.python.keras.api._v2.keras import backend as K

srate = 250
beg = 0.
end = 4.
Samples = int((end - beg) * srate)
K.set_image_data_format('channels_last')

# change these code to visualize your models
if __name__ == '__main__':
    filepath = os.path.join('model', 'example_model.h5')
    model = load_model(filepath, custom_objects={'TSG': TSG})
    model.summary()

    if not model.input.shape[2] == Samples:
        cropping = True
    else:
        cropping = False

    filepath = os.path.join('data', '4s', 'Test', 'example_data' + '.mat')
    vis = visualize(
        model,
        vis_data_file=filepath,
        cropping=cropping,
        # step=25,
        beg=beg,
        end=end,
        srate=srate)

    data = vis._read_data(srate)

    loss, acc = model.evaluate(data['x'], data['y'], batch_size=10, verbose=2)
    _pred = model.predict(data['x'], batch_size=10, verbose=2)
    pred = np.argmax(_pred, axis=1)
    kappa = computeKappa(pred, data['y'])
    print('loss: %.4f\tacc: %.4f\tkappa: %.4f' % (loss, acc, kappa))

    # vis.kernel('tfconv')
    # vis.fft_output('tfconv')
    # vis.fs_fft_output('tfconv')
    # vis.class_fft_output('tfconv')
    # vis.depthwise_kernel('sconv')
    # vis.topo_kernel('sconv')
    # vis.kernel('fs')
    # vis.fs_class_topo_kernel('sconv')
    vis.fs_class_fft_output('tfconv')
    vis.fs_class_freq_topo_kernel('sconv')
    vis.show()