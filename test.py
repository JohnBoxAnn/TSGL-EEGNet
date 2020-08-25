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

# change these code to visualize your models
if __name__ == '__main__':
    data = {'x_test': None, 'y_test': None}
    filepath = os.path.join('data', str(end) + 's', 'Test', 'example_data' + '.mat')
    data['x_test'] = load_data(filepath, label=False)
    data['x_test'] = bandpassfilter(data['x_test'])
    data['x_test'] = data['x_test'][:, :,
                                    int(beg * srate):int(end * srate),
                                    np.newaxis]
    data = crossValidate._normalize(data)
    data['y_test'] = load_data(filepath, label=True)


    filepath = os.path.join('model', 'example_model.h5')
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