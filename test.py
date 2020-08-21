# coding:utf-8
import os
import numpy as np

from core.utils import load_data, filterbank, highpassfilter, bandpassfilter
from core.visualization import visualize
from core.regularizers import TSG
from core.train import crossValidate, create_EEGNet

from tensorflow.python.keras.api._v2.keras.models import load_model
from tensorflow.python.keras.api._v2.keras import backend as K

srate = 250
beg = 0
end = 4
Samples = (end - beg) * srate
K.set_image_data_format('channels_last')

if __name__ == '__main__':
    data = {'x_test': None, 'y_test': None}
    filepath = os.path.join('data', str(end) + 's', 'Test', 'A03E' + '.mat')
    data['x_test'] = load_data(filepath, label=False)
    data['x_test'] = bandpassfilter(data['x_test'])
    data['x_test'] = data['x_test'][:, :,
                                    int(beg * srate):int(end * srate),
                                    np.newaxis]
    data = crossValidate._normalize(data)
    data['y_test'] = load_data(filepath, label=True)

    # EEGNet
    #BEST SUBJECT
    # model = create_EEGNet(4, Samples, F=16, D=10, Ns=20)
    # filepath = os.path.join('model', 'CV_2020_10_17_08_42_30_EEGNet',
    #                         'F(16)_D(10)_Ns(20)_A03T_EEGNet(4).h5')
    #WORST SUBJECT
    # model = create_EEGNet(4, Samples, F=16, D=10, Ns=20)
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
    loss, acc = model.evaluate(data['x_test'],
                               data['y_test'],
                               batch_size=10,
                               verbose=2)
    print('loss: %.4f\tacc: %.4f' % (loss, acc))
    vis = visualize(model, vis_data={'x': data['x_test'], 'y': data['y_test']})
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
    pass