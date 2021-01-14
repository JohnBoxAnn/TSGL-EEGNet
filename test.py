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
    # filepath = os.path.join('model', 'example_model.h5')
    # EEGNet
    #BEST SUBJECT
    # model = create_EEGNet(4, Samples, F=16, D=10, Ns=20, summary=False)
    # filepath = os.path.join('model', 'CV_2020_10_17_08_42_30_EEGNet',
    #                         'F(16)_D(10)_Ns(20)_A03T_EEGNet(4).h5')
    #WORST SUBJECT
    # model = create_EEGNet(4, Samples, F=16, D=10, Ns=20, summary=False)
    # filepath = os.path.join('model', 'CV_2020_10_17_08_42_30_EEGNet',
    #                         'F(16)_D(10)_Ns(20)_A06T_EEGNet(6).h5')
    # model.load_weights(filepath)

    #TSGL-EEGNet
    #BEST SUBJECT
    filepath = os.path.join(
        'model', 'GS_2020_10_16_17_34_36_rawEEGConvNet',
        'l1(0.00010000)_l21(0.00007500)_tl1(0.00000250)_F(16)_D(10)_Ns(20)_FSLength(01)_A03T_rawEEGConvNet(8).h5'
    )
    #WORST SUBJECT
    # filepath = os.path.join(
    #     'model', 'GS_2020_10_16_17_34_36_rawEEGConvNet',
    #     'l1(0.00005000)_l21(0.00010000)_tl1(0.00000100)_F(16)_D(10)_Ns(20)_FSLength(01)_A06T_rawEEGConvNet(10).h5'
    # )
    model = load_model(filepath, custom_objects={'TSG': TSG})
    model.summary()

    if not model.input.shape[2] == Samples:
        cropping = True
    else:
        cropping = False

    filepath = os.path.join('data', 'A', 'Test', 'A03E' + '.mat')
    vis = visualize(
        model,
        vis_data_file=filepath,
        cropping=cropping,
        step=25,
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
    # vis.line_kernel('tfconv', [1,6,7,10,11,15])
    # vis.fft_kernel('tfconv')
    vis.fft_output('tfconv')
    # vis.fs_fft_output('tfconv')
    # vis.class_fft_output('tfconv')
    # vis.depthwise_kernel('sconv')
    # vis.topo_kernel('sconv')
    # vis.fs_class_topo_kernel('sconv')
    # vis.fs_class_fft_output('tfconv')
    # vis.fs_class_freq_topo_kernel('sconv')
    # vis.kernel('fs')
    # vis.line_kernel('fs')
    vis.show()