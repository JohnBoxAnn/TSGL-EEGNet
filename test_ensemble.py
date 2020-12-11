import os
import sys
import json
import numpy as np
import tensorflow as tf

from core.train import crossValidate as _crossValidate
from core.train import create_EEGNet, create_TSGLEEGNet
from core.generators import rawGenerator
from core.splits import StratifiedKFold, AllTrain
from core.regularizers import TSG
from core.utils import computeKappa, walk_files

_console = sys.stdout


class ensembleTest(_crossValidate):
    def __init__(self,
                 built_fn,
                 dataGent,
                 cvfolderpath=None,
                 resultsavepath=None,
                 splitMethod=StratifiedKFold,
                 traindata_filepath=None,
                 testdata_filepath=None,
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
                 standardizing=True,
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 preserve_initfile=False,
                 reinit=True,
                 *args,
                 **kwargs):
        super().__init__(built_fn,
                         dataGent,
                         splitMethod=splitMethod,
                         traindata_filepath=traindata_filepath,
                         testdata_filepath=testdata_filepath,
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
                         standardizing=standardizing,
                         batch_size=batch_size,
                         epochs=epochs,
                         patience=patience,
                         verbose=verbose,
                         preserve_initfile=preserve_initfile,
                         reinit=reinit,
                         *args,
                         **kwargs)

        self.basepath = cvfolderpath
        self.resavepath = resultsavepath
        if not self.resavepath:
            self.resavepath = os.path.join('result', 'ensembleTest.txt')
        self.weight_list = []
        self.ename = 'vote'
        self.weightLearner()

    def weightLearner(self):
        for _ in range(max(self.subs)):
            self.weight_list.append([])
        for sub in self.subs:
            for _ in range(self.kFold):
                self.weight_list[sub - 1].append(1 / self.kFold)

    @staticmethod
    def ensemble(pred_list: list, weight_list: list):
        pred = np.zeros((len(pred_list[0]), 4))
        for week_pred, weight in zip(pred_list, weight_list):
            i = 0
            for p in week_pred:
                pred[i, int(p)] += weight * 1
                i += 1
        return np.argmax(pred, axis=1)

    def call(self, *args, **kwargs):
        gent = self._read_data

        if self.modelstr == 'EEGNet':
            _co = {}
        elif self.modelstr == 'rawEEGConv' or self.modelstr == 'TSGLEEGNet':
            _co = {'TSG': TSG}
        else:
            _co = {}

        acc_list = []
        kappa_list = []
        data = {'x_test': None, 'y_test': None}
        for subject in self.subs:
            pred_list = []
            for path in walk_files(
                    os.path.join(self.basepath, '{:0>2d}'.format(subject))):
                if not self._readed:
                    for data['x_test'], data['y_test'] in gent(subject=subject,
                                                               mode='test'):
                        self._readed = True
                        if self.standardizing:
                            data = self._standardize(data)
                model = tf.keras.models.load_model(path, custom_objects=_co)

                if self.cropping:
                    _Pred = []
                    for cpd in self._cropping_data((data['x_test'], )):
                        pd = model.predict(cpd, verbose=0)
                        _Pred.append(
                            np.argmax(pd, axis=1) == np.squeeze(
                                data['y_test']))
                    _Pred = np.array(_Pred)
                    Pred = []
                    for i in np.arange(_Pred.shape[1]):
                        if _Pred[:, i].any():
                            Pred.append(1)
                        else:
                            Pred.append(0)
                    acc = np.mean(np.array(Pred))
                    kappa = 0.  # None
                else:
                    _pred = model.predict(data['x_test'],
                                          batch_size=self.batch_size,
                                          verbose=self.verbose)
                    pred_list.append(np.squeeze(np.argmax(_pred, axis=1)))
            pred = self.ensemble(pred_list, self.weight_list[subject - 1])
            acc_list.append(np.mean(pred == np.squeeze(data['y_test'])))
            kappa_list.append(computeKappa(pred, data['y_test']))
            self._readed = False
        avg_acc = np.mean(acc_list)
        avg_kappa = np.mean(kappa_list)

        with open(self.resavepath, 'w+') as f:
            sys.stdout = f
            print(('{0:s} {1:d}-fold Ensemble({2:s}) ' + self.validation_name +
                   ' Accuracy (kappa)').format(self.modelstr, self.kFold,
                                               self.ename))
            for i in range(len(self.subs)):
                print('Subject {0:0>2d}: {1:.2%} ({2:.4f})'.format(
                    self.subs[i], acc_list[i], kappa_list[i]))
            print('Average   : {0:.2%} ({1:.4f})'.format(avg_acc, avg_kappa))
            sys.stdout = _console
            f.seek(0, 0)
            for line in f.readlines():
                print(line)
            f.close()
        acc_list.append(avg_acc)
        kappa_list.append(avg_kappa)
        return acc_list, kappa_list

    def getConfig(self):
        config = {'cvfolderpath': self.basepath, 'resavepath': self.resavepath}
        base_config = super(_crossValidate, self).getConfig()
        return dict(list(base_config.items()) + list(config.items()))

    def getSuperConfig(self):
        return super(_crossValidate, self).getConfig()


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
        'dataGent': rawGenerator,
        'splitMethod': AllTrain,
        'cvfolderpath': cvfolderpath,
        'datadir': os.path.join('data', 'C'),
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

    et = ensembleTest(**params)
    avgacc, avgkappa = et()
