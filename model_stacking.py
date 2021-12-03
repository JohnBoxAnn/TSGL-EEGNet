import os
import sys
import json

import numpy as np
import tensorflow as tf

from typing import ClassVar, Callable
from sklearn.linear_model import LinearRegression
from model_ensemble import ensembleTest
from core.get_model import create_EEGNet, create_TSGLEEGNet
from core.dataloaders import RawDataloader
from core.dataloaders import BaseDataloader as _BaseDataloader
from core.generators import RawGenerator
from core.generators import BaseGenerator as _BaseGenerator
from core.splits import StratifiedKFold, AllTrain
from core.splits import _BaseCrossValidator
from core.regularizers import TSG
from core.utils import computeKappa, walk_files

_console = sys.stdout


class stackingTest(ensembleTest):
    def __init__(self,
                 built_fn: Callable[..., tf.keras.Model],
                 dataLoader: _BaseDataloader,
                 dataGent: _BaseGenerator,
                 splitMethod: _BaseCrossValidator = StratifiedKFold,
                 cvfolderpath=None,
                 resultsavepath=None,
                 traindata_filepath=None,
                 testdata_filepath=None,
                 datadir=None,
                 beg=0.0,
                 end=4.0,
                 srate=250,
                 kFold=10,
                 shuffle=False,
                 random_state=None,
                 subs=range(1, 10),
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=25,
                 norm_mode='maxmin',
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 preserve_initfile=False,
                 reinit=True,
                 *args,
                 **kwargs):
        super().__init__(built_fn,
                         dataLoader=dataLoader,
                         dataGent=dataGent,
                         cvfolderpath=cvfolderpath,
                         resultsavepath=resultsavepath,
                         splitMethod=splitMethod,
                         traindata_filepath=traindata_filepath,
                         testdata_filepath=testdata_filepath,
                         datadir=datadir,
                         beg=beg,
                         end=end,
                         srate=srate,
                         kFold=kFold,
                         shuffle=shuffle,
                         random_state=random_state,
                         subs=subs,
                         cropping=cropping,
                         winLength=winLength,
                         cpt=cpt,
                         step=step,
                         norm_mode=norm_mode,
                         batch_size=batch_size,
                         epochs=epochs,
                         patience=patience,
                         verbose=verbose,
                         preserve_initfile=preserve_initfile,
                         reinit=reinit,
                         *args,
                         **kwargs)

        if not resultsavepath:
            self.resavepath = os.path.join('result', 'baggingTest.txt')
        self.ename = 'bagging'

    def weightLearner(self):
        super().weightLearner()
        vrsavepath = os.path.join('result', 'voterate.txt')

        assert (self.cropping == False)

        gent = self._read_data

        if self.modelstr == 'EEGNet':
            _co = {}
        elif self.modelstr == 'rawEEGConv' or self.modelstr == 'TSGLEEGNet':
            _co = {'TSG': TSG}
        else:
            _co = {}

        voterate_list = []
        data = {'x_train': None, 'y_train': None}
        for subject in self.subs:
            pred_list = []
            for path in walk_files(
                    os.path.join(self.basepath, '{:0>2d}'.format(subject)),
                    'h5'):
                if not self._readed:
                    for data['x_train'], data['y_train'] in gent(
                            subject=subject, mode='train'):
                        self._readed = True
                        if self.standardizing:
                            data = self._standardize(data)
                model = tf.keras.models.load_model(path, custom_objects=_co)
                _pred = model.predict(data['x_train'],
                                      batch_size=self.batch_size,
                                      verbose=self.verbose)
                pred_list.append(
                    np.squeeze(
                        np.argmax(_pred, axis=1) == np.squeeze(
                            data['y_train'])))
            pred = np.array(pred_list)
            lr = LinearRegression(fit_intercept=False)
            lr.fit(pred.T, np.squeeze(np.ones_like(data['y_train'])))
            self.weight_list[subject - 1] = lr.coef_
            voterate_list.append(lr.coef_)
            self._readed = False
        with open(vrsavepath, 'w+') as f:
            sys.stdout = f
            print('Bagging Ensemble Vote Rate (Linear Regression)')
            for subject, vr in zip(self.subs, voterate_list):
                print('Subject {:0>2d}: '.format(subject),
                      list(map(lambda x: '{:.2f}'.format(x), vr)))
            sys.stdout = _console
            f.seek(0, 0)
            for line in f.readlines():
                print(line)
            f.close()

    def getConfig(self):
        config = {'cvfolderpath': self.basepath, 'resavepath': self.resavepath}
        base_config = super(ensembleTest, self).getConfig()
        base_config.update(config)
        return base_config


if __name__ == '__main__':
    cvfolderpath = input('Root folder path: ')
    if os.path.exists(cvfolderpath):
        cvfolderpath = os.path.join(cvfolderpath)
    else:
        raise ValueError('Path isn\'t exists.')

    subs = input('Subs (use commas to separate): ').split(',')
    if subs[0][0] == '@':
        subs = [int(subs[0][1:])]
    else:
        subs = list(map(int, subs))
        if len(subs) == 1:
            subs = [i for i in range(1, subs[0] + 1)]
    for i in subs:
        if not os.path.exists(os.path.join(cvfolderpath, '{:0>2d}'.format(i))):
            raise ValueError('subject don\'t exists.')

    params = {
        'built_fn': create_TSGLEEGNet,
        'dataGent': RawGenerator,
        'splitMethod': AllTrain,
        'cvfolderpath': cvfolderpath,
        'datadir': os.path.join('data', 'A'),
        'kFold': 5,
        'subs': subs,
        'cropping': False,
        'standardizing': True
    }

    jsonPath = os.path.join(cvfolderpath, 'params.json')
    if os.path.exists(jsonPath):
        with open(jsonPath, 'r') as f:
            params.update(json.load(f, parse_int=int))
        params['built_fn'] = vars()[params['built_fn']]
        params['dataGent'] = vars()[params['dataGent']]
        params['splitMethod'] = vars()[params['splitMethod']]
        params['subs'] = subs

    bt = stackingTest(**params)
    avgacc, avgkappa = bt()
