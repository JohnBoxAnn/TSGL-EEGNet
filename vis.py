# coding:utf-8
import os
import tensorflow as tf

from core.dataloaders import RawDataloader
from core.generators import RawGenerator, get_steps
from core.models import get_custom_objects, get_compile
from core.get_model import create_TSGLEEGNet
from core.training import crossValidate
from core.metrics import Kappa
from core.visualization import visualize

from tensorflow.python.keras.api._v2.keras.models import load_model
from tensorflow.python.keras.api._v2.keras import backend as K

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
print(gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

srate = 250
beg = 0.
end = 4.
Samples = int((end - beg) * srate)
batch_size = 10
K.set_image_data_format('channels_last')

# change these code to visualize your models
if __name__ == '__main__':

    filepath = os.path.join('model', 'example_model.h5')
    co = get_custom_objects('TSGLEEGNet')
    co.update({'Kappa': Kappa})
    model = load_model(filepath, custom_objects=co)
    model = get_compile(model, metrics=['accuracy', Kappa(4, sparse_labels=True)])
    # model.summary()

    if not model.input.shape[2] == Samples:
        cropping = True
        cpt = model.input.shape[2] / srate
    else:
        cropping = False
        cpt = None

    train_datapath = os.path.join('data', 'A', 'TrainSet', 'example_data.mat')
    test_datapath = os.path.join('data', 'A', 'TestSet', 'example_data.mat')
    datadir = None
    if datadir:
        for root, dirs, files in os.walk(datadir):
            if files:
                dn = files[0][0]
                break
    else:
        dn = ''
    dataloader = RawDataloader(beg=beg,
                               end=end,
                               srate=srate,
                               norm_mode='z-score',
                               traindata_filepath=train_datapath,
                               testdata_filepath=test_datapath,
                               datadir=datadir,
                               dn=dn)
    datagen = RawGenerator(batch_size=10,
                           epochs=1,
                           beg=beg,
                           end=end,
                           srate=srate,
                           cropping=cropping,
                           cpt=cpt,
                           step=int(0.2 * srate),
                           max_crop=1)
    cv = crossValidate(create_TSGLEEGNet,
                       dataLoader=dataloader,
                       dataGent=datagen)
    get_dataset = cv.get_dataset
    cv.subject = 1
    data = cv.get_data()

    loss, acc, kappa = model.evaluate(get_dataset(data['x_test'],
                                                  data['y_test']),
                                      verbose=2,
                                      steps=get_steps(datagen,
                                                      data['x_test'],
                                                      data['y_test'],
                                                      batch_size=batch_size))
    print('Subject 0%d:\tloss: %.4f\tacc: %.4f\tkappa: %.4f' %
          (cv.subject, loss, acc, kappa))
    cv.dataLoader.setReaded(False)

    vis = visualize(model=model,
                    dataGen=datagen,
                    srate=srate,
                    vis_data={
                        'x': data['x_train'],
                        'y': data['y_train']
                    })

    # vis.kernel('tfconv')
    # vis.line_kernel('tfconv', [1,6,7,10,11,15])
    # vis.fft_kernel('tfconv')
    # vis.fft_output('tfconv')
    # vis.fs_fft_output('tfconv')
    # vis.class_fft_output('tfconv')
    # vis.depthwise_kernel('sconv')
    # vis.topo_kernel('sconv')
    # vis.fs_class_topo_kernel('sconv')
    vis.fs_class_fft_output('tfconv')
    # vis.fs_class_freq_topo_kernel('sconv')
    # vis.kernel('fs')
    # vis.line_kernel('fs')
    vis.show()