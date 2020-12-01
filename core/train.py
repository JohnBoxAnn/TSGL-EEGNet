# coding:utf-8

import os
import gc
import sys
import math
import copy
import time
import logging
import itertools
import numpy as np
from numpy.core.numeric import cross
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.keras.api._v2.keras import backend as K

from core.models import EEGNet, TSGLEEGNet, ShallowConvNet, DeepConvNet, MB3DCNN
from core.splits import StratifiedKFold
from core.callbacks import MyModelCheckpoint, EarlyStopping
from core.utils import standardization, computeKappa

_console = sys.stdout


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
    model.compile(optimizer=optimizer(lrate), loss=loss, metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, 'MB3DCNN.png', show_shapes=True)
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
    model.compile(optimizer=optimizer(lrate), loss=loss, metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, 'EEGNet.png', show_shapes=True)
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
    model.compile(optimizer=optimizer(lrate), loss=loss, metrics=metrics)
    if summary:
        model.summary()
        # export graph of the model
        # tf.keras.utils.plot_model(model, 'rawEEGConvNet.png', show_shapes=True)
    return model


class crossValidate(object):
    '''
    Class for K-fold Cross Validation.

    This framework can collect `model`, `loss`, `acc` and `history` from each fold, and 
    save them into files. 
    Data spliting methods from sklearn.model_selection are supported. you can pass the 
    classes as `splitMethod`. 

    This class has implemented a magic method `__call__()` wrapping `call()`, for which
     it can be used like a function.

    Parameters
    ----------
    ```txt
    built_fn        : function, Create Training model which need to cross-validate.
                      Please using string `create_` at the begining of function name, 
                      like `create_modelname`.
    dataGent        : class, Generate data for @built_fn, shapes (n_trails, ...). 
                      It should discriminate data and label.
                      More details see core.generators.
    splitMethod     : class, Support split methods from module sklearn.model_selection.
    kFold           : int, Number of K-fold.
    shuffle         : bool, Optional Whether to shuffle each class's samples before 
                      splitting into batches, default = False.
    random_state    : int, RandomState instance or None, optional, default = None. 
                      If int, random_state is the seed used by the random number 
                      generator; If RandomState instance, random_state is the random 
                      number generator; If None, the random number generator is the 
                      RandomState instance used by np.random. Used when shuffle == True.
    subs            : list, list of subjects' number, like `range(1, 10)`.
    cropping        : bool, Switch of cropped training. Default = False.
    winLength       : int, cropping window length, default = 2*srate.
    cpt             : float, cropping sencond, optional, only available when `winLength`
                      is not specified.
    step            : int, cropping step, default = 4.
    standardizing   : bool, Switch of standardizing data. Default = True.
    batch_size      : int, Batch size.
    epochs          : int, Training epochs.
    patience        : int, Early stopping patience.
    verbose         : int, One of 0, 1 and 2.
    *a, *args       : tuple, Parameters used by @dataGent and @built_fn respectively
    **kw, **kwargs  : dict, Parameters used by @dataGent and @built_fn respectively, 
                      **kw should include parameters called `beg`, `end` and `srate`.
    ```

    Returns
    -------
    ```txt
    avg_acc         : list, Average accuracy for each subject with K-fold Cross Validation, 
                      and total average accuracy is at the last of the list
    avg_kappa       : list, Average kappa for each subject with K-fold Cross Validation, 
                      and total average kappa is at the last of the list
    ```

    Example
    -------
    ```python
    from core.splits import StratifiedKFold

    def create_model(Samples, *args, summary=True, **kwargs):
        ...
        return keras_model

    class dataGenerator:
        def __init__(self, *a, beg=0, end=4, srate=250, **kw):
            ...

        def __call__(self, filepath, label=False):
            if label:
                ...
                return label
            else:
                ...
                return data
        ...
    ...
    avg_acc = crossValidate(
                create_model, 
                dataGenerator, 
                beg=0,
                end=4,
                srate=250,
                splitMethod=StratifiedKFold,
                kFold=10, 
                subs=range(1, 10), 
                *a, 
                **kw)(*args, **kwargs)
    ```

    Note
    ----
    More details to see the codes.
    '''
    def __init__(self,
                 built_fn,
                 dataGent,
                 splitMethod=StratifiedKFold,
                 traindata_filepath=None,
                 testdata_filepath=None,
                 datadir=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 kFold=10,
                 shuffle=False,
                 random_state=None,
                 subs: list = range(1, 10),
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
        self.built_fn = built_fn
        self.dataGent = dataGent(beg=beg,
                                 end=end,
                                 srate=srate,
                                 *args,
                                 **kwargs)
        self.beg = beg
        self.end = end
        self.srate = srate
        self.splitMethod = splitMethod
        self.traindata_filepath = traindata_filepath
        self.testdata_filepath = testdata_filepath
        self.datadir = datadir
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.subs = subs
        self.cropping = cropping
        self.winLength = winLength
        self.cpt = cpt
        self.step = step
        self.standardizing = standardizing
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.preserve_initfile = preserve_initfile
        self.reinit = reinit
        self.args = args
        self.kwargs = kwargs
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self._check_params()

        if self.datadir:
            for root, dirs, files in os.walk(self.datadir):
                if files:
                    self.dn = files[0][0]
                    break

        self.modelstr = built_fn.__name__[7:]
        if self.splitMethod.__name__ == 'AllTrain':
            self.validation_name = 'Average Validation'
        else:
            self.validation_name = 'Cross Validation'
        self._new_fold = True
        self._last_batch = False

        self._readed = False
        self.X1 = None
        self.Y1 = None
        self.X2 = None
        self.Y2 = None

        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('result'):
            os.makedirs('result')

        # cropped training
        if self.winLength:
            if not isinstance(self.winLength, int):
                raise TypeError('`winLength` must be passed as int.')
            if self.winLength > (self.end - self.beg) * self.srate:
                raise ValueError(
                    '`winLength` must less than or equal (`end` - '
                    '`beg`) * `srate`.')
        if self.cpt and not self.winLength:
            if (isinstance(self.cpt, float) or isinstance(self.cpt, int)):
                if self.cpt <= self.end - self.beg:
                    self.winLength = self.cpt * self.srate
                else:
                    raise ValueError(
                        '`cpt` must less than or equal `end` - `beg`.')
            else:
                raise TypeError('`cpt` must be passed as int or float.')
        if not self.winLength:
            self.winLength = 2 * self.srate
        if self.step:
            if not isinstance(self.step, int):
                raise TypeError('`step` must be passed as int.')
        else:
            self.step = 4

    def call(self, *args, **kwargs):
        initfile = os.path.join('.', 'CV_initweight.h5')
        name = 'Cross Validate'
        tm = time.localtime()
        dirname = (
            'CV_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_{6:s}'.
            format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                   tm.tm_sec, self.modelstr))
        if not os.path.exists(os.path.join('model', dirname)):
            os.mkdir(os.path.join('model', dirname))
        if not os.path.exists(os.path.join('result', dirname)):
            os.mkdir(os.path.join('result', dirname))

        if self.cropping:
            gent = self._gent_cropped_data
            self.Samples -= self.winLength
        else:
            gent = self._gent_data

        if not self.reinit:
            model = self.built_fn(*args, **kwargs, Samples=self.Samples)
            model.save_weights(initfile)

        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=self.patience,
                                      verbose=0,
                                      mode='auto')

        filename = ''
        for key in kwargs.keys():
            if key in ['l1', 'l21', 'tl1']:
                filename += '{0:s}({1:.8f})_'.format(key, kwargs[key])
            else:
                filename += '{0:s}({1:0>2d})_'.format(key, kwargs[key])

        avg_acci = []
        avg_kappai = []
        win_subs_list = []
        for i in self.subs:
            accik = []
            kappaik = []
            k = 0  # count kFolds
            t = 0  # record model's saving time
            c = 0  # count windows
            win = 0  # selected windows
            win_list = []  # selected windows list
            for data in gent(subject=i):
                if self._new_fold:  # new fold for cropped training
                    self._new_fold = False
                    c = 0

                    if self.reinit:
                        model = self.built_fn(*args,
                                              **kwargs,
                                              Samples=self.Samples)

                    k += 1
                    filepath = os.path.join(
                        'result', dirname,
                        filename + '{:s}.txt'.format(self.modelstr))
                    with open(filepath, 'w+') as f:
                        sys.stdout = f
                        print(('{0:s} {1:d}-fold ' + self.validation_name +
                               ' Accuracy').format(self.modelstr, self.kFold))
                        print('Subject {:0>2d} fold {:0>2d} in processing'.
                              format(i, k))
                        sys.stdout = _console
                        f.seek(0, 0)
                        for line in f.readlines():
                            print(line)
                        f.close()

                    filepath = os.path.join(
                        'model', dirname, filename +
                        'A0{0:d}T_{1:s}({2:d}).h5'.format(i, self.modelstr, k))
                    checkpointer = MyModelCheckpoint(filepath=filepath,
                                                     verbose=1,
                                                     save_best_only=True,
                                                     statistic_best=True,
                                                     p=0.05)
                    history = {}
                else:
                    c += 1

                # TODO: fit(), evaluate() with tf.data.Dataset, then `self._new_fold`
                #       and `self._last_batch` will be DEPRECATED.
                history = dict(
                    list(history.items()) + list(
                        model.fit(x=data['x_train'],
                                  y=data['y_train'],
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  callbacks=[checkpointer, earlystopping],
                                  verbose=self.verbose,
                                  validation_data=[
                                      data['x_val'], data['y_val']
                                  ]).history.items()))

                if self.cropping:
                    if not t == os.path.getmtime(checkpointer._filepath):
                        t = os.path.getmtime(checkpointer._filepath)
                        win = c

                # tf.keras.models.Model.fit()
                # tf.keras.models.Model.evaluate()
                # tf.data.Dataset.from_generator()

                # load the best model for cropped training or evaluating its accuracy
                model.load_weights(filepath)

                if self._last_batch:  # the last batch for cropped training
                    self._last_batch = False

                    if self.cropping:
                        win_list.append(win)
                        print(win_list)
                        x_test = data['x_test'][:, :, win *
                                                self.step:win * self.step +
                                                self.Samples, :]
                        pd = model.predict(x_test, verbose=0)
                        pred = np.argmax(pd, axis=1)
                        acc = np.mean(
                            np.squeeze(pred) == np.squeeze(data['y_test']))
                        kappa = computeKappa(pred, data['y_test'])
                        print('acc: {:.2%}\nkappa: {:.4f}'.format(acc, kappa))
                    else:
                        loss, acc = model.evaluate(data['x_test'],
                                                   data['y_test'],
                                                   batch_size=self.batch_size,
                                                   verbose=self.verbose)
                        _pred = model.predict(data['x_test'],
                                              batch_size=self.batch_size,
                                              verbose=self.verbose)
                        pred = np.argmax(_pred, axis=1)
                        kappa = computeKappa(pred, data['y_test'])

                    # save the train history
                    filepath = filepath[:-3] + '.npy'
                    np.save(filepath, history)

                    # reset model's weights to train a new one next fold
                    if os.path.exists(initfile) and not self.reinit:
                        model.load_weights(initfile)
                        model.reset_states()

                    if self.reinit:
                        K.clear_session()
                        del model
                        gc.collect()

                    accik.append(acc)
                    kappaik.append(kappa)
            avg_acci.append(np.average(np.array(accik)))
            avg_kappai.append(np.average(np.array(kappaik)))
            win_subs_list.append(win_list)
            self._readed = False
        avg_acc = np.average(np.array(avg_acci))
        avg_kappa = np.average(np.array(avg_kappai))
        filepath = os.path.join(
            'result',
            'CV_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_' \
            '{6:s}.txt'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour,
                               tm.tm_min, tm.tm_sec, self.modelstr))
        with open(filepath, 'w+') as f:
            sys.stdout = f
            print(('{0:s} {1:d}-fold ' + self.validation_name +
                   ' Accuracy (kappa)').format(self.modelstr, self.kFold))
            for i in range(len(self.subs)):
                print('Subject {0:0>2d}: {1:.2%} ({2:.4f})'.format(
                    self.subs[i], avg_acci[i], avg_kappai[i]))
            print('Average   : {0:.2%} ({1:.4f})'.format(avg_acc, avg_kappa))
            sys.stdout = _console
            f.seek(0, 0)
            for line in f.readlines():
                print(line)
            f.close()
        if os.path.exists(initfile) and not self.preserve_initfile:
            os.remove(initfile)
        avg_acci.append(avg_acc)
        avg_kappai.append(avg_kappa)
        return avg_acci, avg_kappai

    def __call__(self, *args, **kwargs):
        '''Wraps `call()`.'''
        return self.call(*args, **kwargs)

    def getConfig(self):
        config = {
            'built_fn': self.built_fn,
            'dataGent': self.dataGent,
            'splitMethod': self.splitMethod,
            'traindata_filepath': self.traindata_filepath,
            'testdata_filepath': self.testdata_filepath,
            'datadir': self.datadir,
            'beg': self.beg,
            'end': self.end,
            'srate': self.srate,
            'kFold': self.kFold,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
            'subs': self.subs,
            'cropping': self.cropping,
            'winLength': self.winLength,
            'step': self.step,
            'standardizing': self.standardizing,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'patience': self.patience,
            'verbose': self.verbose,
            'preserve_initfile': self.preserve_initfile,
            'reinit': self.reinit,
            'args': self.args,
            'kwargs': self.kwargs
        }
        return config

    def setConfig(self,
                  built_fn,
                  dataGent,
                  splitMethod=StratifiedKFold,
                  traindata_filepath=None,
                  testdata_filepath=None,
                  datadir=None,
                  beg=0.,
                  end=4.,
                  srate=250,
                  kFold=10,
                  shuffle=False,
                  random_state=None,
                  subs: list = range(1, 10),
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
        self.built_fn = built_fn
        self.dataGent = dataGent(*args,
                                 **kwargs,
                                 beg=beg,
                                 end=end,
                                 srate=srate)
        self.beg = beg
        self.end = end
        self.srate = srate
        self.splitMethod = splitMethod
        self.traindata_filepath = traindata_filepath
        self.testdata_filepath = testdata_filepath
        self.datadir = datadir
        self.kFold = kFold
        self.shuffle = shuffle
        self.random_state = random_state
        self.subs = subs
        self.cropping = cropping
        self.winLength = winLength
        self.step = step
        self.standardizing = standardizing
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.args = args
        self.kwargs = kwargs
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)
        self._check_params()

        self.modelstr = built_fn.__name__[7:]
        if self.dataGent.__name__ == 'AllTrain':
            self.validation_name = 'Average Validation'
        else:
            self.validation_name = 'Cross Validation'
        self.built = False
        self._new_fold = True
        self._last_batch = False

        self._readed = False
        self.X1 = None
        self.Y1 = None
        self.X2 = None
        self.Y2 = None

        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('result'):
            os.makedirs('result')

        # cropped training
        if self.winLength:
            if not isinstance(self.winLength, int):
                raise TypeError('`winLength` must be passed as int.')
            if self.winLength > (self.end - self.beg) * self.srate:
                raise ValueError(
                    '`winLength` must less than or equal (`end` - '
                    '`beg`) * `srate`.')
        if self.cpt and not self.winLength:
            if (isinstance(self.cpt, float) or isinstance(self.cpt, int)):
                if self.cpt <= self.end - self.beg:
                    self.winLength = self.cpt * self.srate
                else:
                    raise ValueError(
                        '`cpt` must less than or equal `end` - `beg`.')
            else:
                raise TypeError('`cpt` must be passed as int or float.')
        if not self.winLength:
            self.winLength = 2 * self.srate
        if self.step:
            if not isinstance(self.step, int):
                raise TypeError('`step` must be passed as int.')
        else:
            self.step = 4

    @staticmethod
    def _standardize(data: dict, trialaxis=0):
        '''Standardizing (z-score) on each trial, supports np.nan numbers'''
        # suppose every trials are independent to each other
        meta = ['x_train', 'x_test', 'x_val']

        # to prevent different objects be the same one
        data = copy.deepcopy(data)

        for s in meta:
            if s in data and not data[s] is None:
                _len = len(data[s].shape)
                if _len > 1:
                    axis = list(range(_len))
                    axis.pop(trialaxis)
                    axis = tuple(axis)
                else:
                    axis = -1
                # z-score on trials
                data[s] = standardization(data[s], axis=axis)

        return data

    def _read_data(self, subject, mode):
        '''
        Read data from dataGent.

        Parameters
        ----------
        ```txt
        subject : int, Identifier of subject.
        mode    : str, One of 'train' and 'test'.
        ```

        Yields
        ------
        ```txt
        data    : tuple, (x, y).
        ```
        '''
        meta = ['train', 'test']
        if not isinstance(mode, str):
            raise TypeError('`mode` must be passed as string.')
        if not mode in meta:
            raise ValueError('`mode` must be one of \'train\' and \'test\'.')
        if mode == 'test':
            if not self.testdata_filepath:
                self.testdata_filepath = os.path.join(
                    self.datadir, 'Test',
                    self.dn + '0' + str(subject) + 'E.mat')
                yield self.dataGent(self.testdata_filepath)
                self.testdata_filepath = None
            else:
                yield self.dataGent(self.testdata_filepath)
        else:
            if not self.traindata_filepath:
                self.traindata_filepath = os.path.join(
                    self.datadir, 'Train',
                    self.dn + '0' + str(subject) + 'T.mat')
                yield self.dataGent(self.traindata_filepath)
                self.traindata_filepath = None
            else:
                yield self.dataGent(self.traindata_filepath)

    def _gent_data(self, subject):
        '''
        Generate (data, label) from dataGent.

        Parameters
        ----------
        ```txt
        subject     : int, Identifier of subject.
        ```

        Yields
        ------
        ```txt
        data        : dict, Includes train, val and test data.
        ```
        '''
        data = {
            'x_train': None,
            'y_train': None,
            'x_val': None,
            'y_val': None,
            'x_test': None,
            'y_test': None
        }
        if not self._readed:
            # for once
            for (self.X1, self.Y1) in self._read_data(subject=subject,
                                                      mode='test'):
                pass
            for (self.X2, self.Y2) in self._read_data(subject=subject,
                                                      mode='train'):
                self._readed = True
        data['x_test'] = self.X1
        data['y_test'] = self.Y1
        # for multiple times
        for (x1, y1), (x2, y2) in self._spilt(self.X2, self.Y2):
            data['x_train'] = x1
            data['y_train'] = y1
            data['x_val'] = x2
            data['y_val'] = y2
            if self.standardizing:
                data = self._standardize(data)
            if data['x_val'] is None:
                data['x_val'] = data['x_test']
                data['y_val'] = data['y_test']
            self._new_fold = True
            self._last_batch = True
            yield data

    def _gent_cropped_data(self, subject):
        '''
        Generate cropped (data, label) from dataGent.

        Not including test data.

        Parameters
        ----------
        ```txt
        subject     : int, Identifier of subject.
        ```
        
        Yields
        ------
        ```txt
        data        : dict, Includes train, val and test data.
        ```
        '''
        temp = {
            'x_train': None,
            'y_train': None,
            'x_val': None,
            'y_val': None,
            'x_test': None,
            'y_test': None
        }
        L = range(0, self.Samples + 1, self.step)
        L = len(L)
        print('len(L): {0:d}'.format(L))

        if not self._readed:
            # for once
            for (self.X1, self.Y1) in self._read_data(subject=subject,
                                                      mode='test'):
                pass
            for (self.X2, self.Y2) in self._read_data(subject=subject,
                                                      mode='train'):
                self._readed = True

        temp['x_test'] = self.X1
        temp['y_test'] = self.Y1
        for (x1, y1), (x2, y2) in self._spilt(self.X2, self.Y2):
            temp['x_train'] = x1
            temp['y_train'] = y1
            temp['x_val'] = x2
            temp['y_val'] = y2
            if temp['x_val'] is None:
                if self.standardizing:
                    data = self._standardize(temp)
                    temp['x_val'] = data['x_test']
                    temp['y_val'] = data['y_test']
                    temp['x_test'] = data['x_test']
                    temp['y_test'] = data['y_test']
                else:
                    data['x_train'] = x1
                    data['x_val'] = temp['x_val']
            else:
                if self.standardizing:
                    data = self._standardize(temp)
                    temp['x_test'] = data['x_test']
                    temp['y_test'] = data['y_test']
                else:
                    data['x_train'] = x1
                    data['x_val'] = x2

            i = 0
            for (temp['x_train'], temp['x_val']) in self._cropping_data(
                (data['x_train'], data['x_val'])):
                i += 1
                if i == 1:
                    self._new_fold = True
                if i == L:
                    self._last_batch = True
                yield temp

    def _cropping_data(self, datas: tuple):
        L = range(0, self.Samples + 1, self.step)
        for i in L:
            temp = ()
            for data in datas:
                temp += (data[:, :, i:i + self.winLength, :], )
            yield temp

    def _spilt(self, X, y, groups=None):
        """
        Generate indices to split data into training and test set.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape (n_samples,)
            The target variable for supervised learning problems.

        groups : array-like, with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set. Action depends on the split method you choose.

        Yields
        ------
        train : ndarray
            The training set indices for that split.

        val : ndarray
            The validating set indices for that split.
        """
        sm = self.splitMethod(n_splits=self.kFold,
                              shuffle=self.shuffle,
                              random_state=self.random_state)
        for train_index, val_index in sm.split(X, y, groups):
            # (x_train, y_train), (x_val, y_val)
            if not train_index.any():
                raise ValueError('Training data shouldn\'t be empty.')
            elif not val_index.any():
                yield (X[train_index], y[train_index]), (None, None)
            else:
                yield (X[train_index], y[train_index]), (X[val_index],
                                                         y[val_index])

    def _check_params(self):
        '''
        Cross Validate check parameters out.
        '''
        # TODO: check parameters out.
        pass


class gridSearch(crossValidate):
    '''
    Class for K-fold Cross Validation Grid Search.

    Grid Search method. May better to be a subclass of `crossValidate`. 

    This framework can collect `model`, `loss`, `acc` and `history` from each fold, and 
    save them into files. 
    Data spliting methods from sklearn.model_selection are supported. you can pass the 
    classes as `splitMethod`. 

    It can't use multiple GPUs to speed up now. To grid search on a large parameter 
    matrix, you should use `Greedy Algorithm`.

    This class has implemented a magic method `__call__()` wrapping `call()`, for which
     it can be used like a function.

    Parameters
    ----------
    ```txt
    built_fn        : function, Create Training model which need to cross-validate.
                      Please using string `create_` at the begining of function name, 
                      like `create_modelname`.
    parameters      : dict, Parameters need to grid-search. Keys are the parameters' 
                      name, and every parameter values are vectors which should be 
                      passed as a list.
    dataGent        : class, Generate data for @built_fn, shapes (n_trails, ...). 
                      It should discriminate data and label.
                      More details see core.generators.
    splitMethod     : class, Support split methods from module sklearn.model_selection.
    kFold           : int, Number of K-fold.
    shuffle         : bool, Optional Whether to shuffle each class's samples before 
                      splitting into batches, default = False.
    random_state    : int, RandomState instance or None, optional, default = None. 
                      If int, random_state is the seed used by the random number 
                      generator; If RandomState instance, random_state is the random 
                      number generator; If None, the random number generator is the 
                      RandomState instance used by np.random. Used when shuffle == True.
    subs            : list, list of subjects' number, like `range(1, 10)`.
    cropping        : bool, Switch of cropped training. Default = False.
    winLength       : int, cropping window length, default = 2*srate.
    step            : int, cropping step, default = 1.
    standardizing   : bool, Switch of standardizing data. Default = True.
    batch_size      : int, Batch size.
    epochs          : int, Training epochs.
    patience        : int, Early stopping patience.
    verbose         : int, One of 0, 1 and 2.
    *a, *args       : tuple, Parameters used by @dataGent and @built_fn respectively
    **kw, **kwargs  : dict, Parameters used by @dataGent and @built_fn respectively, 
                      **kw should include parameters called `beg`, `end` and `srate`.
    ```

    Returns
    -------
    ```txt
    avg_acc         : list, Average accuracy for each subject with K-fold Cross Validation, 
                      and total average accuracy is at the last of the list
    avg_kappa       : list, Average kappa for each subject with K-fold Cross Validation, 
                      and total average kappa is at the last of the list
    ```

    Example
    -------
    ```python
    from core.splits import StratifiedKFold

    def create_model(Samples, *args, summary=True, **kwargs):
        ...
        return keras_model

    class dataGenerator:
        def __init__(self, *a, beg=0, end=4, srate=250, **kw):
            ...

        def __call__(self, filepath, label=False):
            if label:
                ...
                return label
            else:
                ...
                return data
        ...
    ...
    parameters = {'para1':[...], 'para2':[...], ...}
    avg_acc = gridSearch(
                create_model, 
                parameters,
                dataGenerator, 
                beg=0,
                end=4,
                srate=250,
                splitMethod=StratifiedKFold,
                kFold=10, 
                subs=range(1, 10), 
                *a, 
                **kw)(*args, **kwargs)
    ```

    Note
    ----
    More details to see the codes.
    '''
    def __init__(self,
                 built_fn,
                 parameters: dict,
                 dataGent,
                 splitMethod=StratifiedKFold,
                 beg=0,
                 end=4,
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
                 reinit=False,
                 *args,
                 **kwargs):
        super().__init__(built_fn=built_fn,
                         dataGent=dataGent,
                         splitMethod=splitMethod,
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
        _subs_targeted = False
        _subs_targeted_parameters = []
        for parameter in parameters:
            if not parameter in self.built_fn.__code__.co_varnames:
                raise ValueError('`parameters` has unsupported parameter in'
                                 ' `built_fn`.')
            if not isinstance(parameters[parameter], list) and not isinstance(
                    parameters[parameter], dict):
                parameters[parameter] = list(parameters[parameter])
            if isinstance(parameters[parameter], dict):
                subs = list(self.subs)
                for subject in parameters[parameter]:
                    if not int(subject) in self.subs:
                        raise ValueError('`parameters` has unsolved subject'
                                         ' numbers.')
                    if not isinstance(parameters[parameter][subject], list):
                        parameters[parameter][subject] = list(
                            parameters[parameter][subject])
                    subs.remove(int(subject))
                if subs:
                    raise ValueError('`parameters` doesn\'t include all the'
                                     ' subject numbers.')
                _subs_targeted = True
                _subs_targeted_parameters.append(parameter)
        temp = []
        if _subs_targeted:
            for subject in self.subs:
                items = []
                for parameter in parameters:
                    if parameter in _subs_targeted_parameters:
                        items += list(
                            {parameter:
                             parameters[parameter][str(subject)]}.items())
                    else:
                        items += list({parameter:
                                       parameters[parameter]}.items())
                temp.append(dict(items))
        else:
            for subject in self.subs:
                temp.append(parameters)

        self.parameters = temp

    def call(self, *args, **kwargs):
        '''
        parameters should be lists to different subjects, then pass one 
        subject's parameter to cv.
        '''
        initfile = os.path.join('.', 'GSCV_initweight.h5')
        name = 'Cross Validate Grid Search'
        tm = time.localtime()
        dirname = (
            'GS_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_{6:s}'.
            format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                   tm.tm_sec, self.modelstr))
        if not os.path.exists(os.path.join('model', dirname)):
            os.mkdir(os.path.join('model', dirname))
        if not os.path.exists(os.path.join('result', dirname)):
            os.mkdir(os.path.join('result', dirname))

        if self.cropping:
            gent = self._gent_cropped_data
            self.Samples -= self.winLength
        else:
            gent = self._gent_data

        earlystopping = EarlyStopping(monitor='val_loss',
                                      min_delta=0,
                                      patience=self.patience,
                                      verbose=0,
                                      mode='auto')

        def cv(*args, **kwargs):
            '''one subject, one parameter'''
            if not self.reinit:
                if not os.path.exists(initfile):
                    model = self.built_fn(*args,
                                          **kwargs,
                                          Samples=self.Samples)
                    model.save_weights(initfile)
                else:
                    model = self.built_fn(*args,
                                          **kwargs,
                                          Samples=self.Samples)
                    model.load_weights(initfile)

            filename = ''
            for key in kwargs.keys():
                if key in ['l1', 'l21', 'tl1']:
                    filename += '{0:s}({1:.8f})_'.format(key, kwargs[key])
                else:
                    filename += '{0:s}({1:0>2d})_'.format(key, kwargs[key])

            acck = []
            kappak = []
            k = 0
            for data in gent(subject=self.subs):
                if self._new_fold:  # new fold for cropped training
                    self._new_fold = False

                    if self.reinit:
                        model = self.built_fn(*args,
                                              **kwargs,
                                              Samples=self.Samples)

                    k += 1
                    filepath = os.path.join(
                        'model', dirname,
                        filename + 'A0{0:d}T_{1:s}({2:d}).h5'.format(
                            self.subs, self.modelstr, k))
                    checkpointer = MyModelCheckpoint(
                        filepath=filepath,
                        verbose=1,
                        #  save_weights_only=True, # save model occures error
                        save_best_only=True,
                        statistic_best=True,
                        p=0.05)
                    history = {}

                # TODO: fit(), evaluate() with tf.data.Dataset, then `self._new_fold`
                #       and `self._last_batch` will be DEPRECATED.
                history = dict(
                    list(history.items()) + list(
                        model.fit(x=data['x_train'],
                                  y=data['y_train'],
                                  batch_size=self.batch_size,
                                  epochs=self.epochs,
                                  callbacks=[checkpointer, earlystopping],
                                  verbose=self.verbose,
                                  validation_data=[
                                      data['x_val'], data['y_val']
                                  ]).history.items()))

                # load the best model for cropped training or evaluating its accuracy
                model.load_weights(filepath)

                if self._last_batch:  # the last batch for cropped training
                    self._last_batch = False

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
                        print('acc: {:.2%}'.format(acc))
                    else:
                        loss, acc = model.evaluate(data['x_test'],
                                                   data['y_test'],
                                                   batch_size=self.batch_size,
                                                   verbose=self.verbose)
                        _pred = model.predict(data['x_test'],
                                              batch_size=self.batch_size,
                                              verbose=self.verbose)
                        pred = np.argmax(_pred, axis=1)
                        kappa = computeKappa(pred, data['y_test'])

                    # save the train history
                    npy_filepath = filepath[:-3] + '.npy'
                    np.save(npy_filepath, history)

                    # reset model's weights to train a new one next fold
                    if os.path.exists(initfile) and not self.reinit:
                        model.load_weights(initfile)
                        model.reset_states()

                    if self.reinit:
                        K.clear_session()
                        del model
                        gc.collect()

                    acck.append(acc)
                    kappak.append(kappa)

            data.clear()
            del data

            K.clear_session()
            del model
            gc.collect()

            avg_acc = np.average(np.array(acck))
            avg_kappa = np.average(np.array(kappak))
            filepath = os.path.join(
                'result', dirname, filename +
                'A0{0:d}T_{1:s}.txt'.format(self.subs, self.modelstr))
            with open(filepath, 'w+') as f:
                sys.stdout = f
                print(('{0:s} {1:d}-fold ' + self.validation_name +
                       ' Accuracy').format(self.modelstr, self.kFold))
                print('Subject {0:0>2d}'.format(self.subs))
                for i in range(len(acck)):
                    print('Fold {0:0>2d}: {1:.2%} ({2:.4f})'.format(
                        i + 1, acck[i], kappak[i]))
                print('Average   : {0:.2%} ({1:.4f})'.format(
                    avg_acc, avg_kappa))
                sys.stdout = _console
                f.seek(0, 0)
                for line in f.readlines():
                    print(line)
                f.close()

            return avg_acc, avg_kappa

        parameters = []
        max_avg_acc = []
        max_acc_kappa = []
        indices = []
        subs = copy.copy(self.subs)
        filepath = os.path.join(
            'result',
            'GS_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_' \
            '{6:s}.txt'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour,
                               tm.tm_min, tm.tm_sec, self.modelstr))
        for subject in subs:
            parameters.append(self._combination(subject=subject))
            count = 0
            with open(filepath, 'w+') as f:
                sys.stdout = f
                print('Subject: {0:0>2d}/{1:0>2d}'.format(subject, len(subs)))
                print(
                    'Grid Search progress: {0:0>4d}/{1:0>4d}' \
                    '\nThe No.{2:0>4d} is in processing'
                    .format(count, len(parameters[-1]), count + 1))
                sys.stdout = _console
                f.seek(0, 0)
                for line in f.readlines():
                    print(line)
                f.close()
            avg_acc = []
            avg_kappa = []
            for parameter in parameters[-1]:
                self.subs = subject
                param = dict(parameter + list(kwargs.items()))
                acc, kappa = cv(*args, **param)
                avg_acc.append(acc)
                avg_kappa.append(kappa)
                count += 1
                with open(filepath, 'w+') as f:
                    sys.stdout = f
                    print('Subject: {0:0>2d}/{1:0>2d}'.format(
                        subject, len(subs)))
                    if count < len(parameters[-1]):
                        print(
                            'Grid Search progress: {0:0>4d}/{1:0>4d}' \
                            '\nThe No.{2:0>4d} is in processing'
                            .format(count, len(parameters[-1]), count + 1))
                    else:
                        print('Grid Search progress: {0:0>4d}/{1:0>4d}'.format(
                            count, len(parameters[-1])))
                    sys.stdout = _console
                    f.seek(0, 0)
                    for line in f.readlines():
                        print(line)
                    f.close()
            self._readed = False
            max_avg_acc.append(np.max(avg_acc))
            indices.append(np.argmax(avg_acc))
            max_acc_kappa.append(avg_kappa[indices[-1]])
        self.subs = subs
        if os.path.exists(initfile) and not self.preserve_initfile:
            os.remove(initfile)

        with open(filepath, 'w+') as f:
            sys.stdout = f
            print(('{0:s} {1:d}-fold ' + name + ' Accuracy (kappa)').format(
                self.modelstr, self.kFold))
            for i in range(len(self.subs)):
                print('Subject {0:0>2d}: {1:.2%} ({2:.4f})'.format(
                    self.subs[i], max_avg_acc[i], max_acc_kappa[i]))
                print('Parameters', end='')
                for n in range(len(parameters[i][indices[i]])):
                    if n == 0:
                        print(': {0:s} = {1:.8f}'.format(
                            parameters[i][indices[i]][n][0],
                            parameters[i][indices[i]][n][1]),
                              end='')
                    else:
                        print(', {0:s} = {1:.8f}'.format(
                            parameters[i][indices[i]][n][0],
                            parameters[i][indices[i]][n][1]),
                              end='')
                print()
            print('Average   : {:.2%} ({:.4f})'.format(
                np.average(max_avg_acc), np.average(max_acc_kappa)))
            sys.stdout = _console
            f.seek(0, 0)
            for line in f.readlines():
                print(line)
            f.close()
        avg_acc = max_avg_acc
        avg_kappa = max_acc_kappa
        avg_acc.append(np.average(max_avg_acc))
        avg_kappa.append(np.average(max_acc_kappa))
        return avg_acc, avg_kappa

    def _combination(self, subject):
        '''Solve the combaination of parameters given to Grid Search'''
        parameters = []
        parameter = []
        keys = list(self.parameters[subject - 1].keys())
        values = list(itertools.product(*self.parameters[subject -
                                                         1].values()))

        for v in values:
            for i in range(len(v)):
                parameter.append((keys[i], v[i]))
            parameters.append(parameter)
            parameter = []

        return parameters

    def getConfig(self):
        config = {'parameters': self.parameters}
        base_config = super(crossValidate, self).getConfig()
        return dict(list(base_config.items()) + list(config.items()))

    def getSuperConfig(self):
        return super(crossValidate, self).getConfig()

    def setConfig(self,
                  built_fn,
                  parameters: dict,
                  dataGent,
                  splitMethod=StratifiedKFold,
                  beg=0,
                  end=4,
                  srate=250,
                  kFold=10,
                  shuffle=False,
                  random_state=None,
                  subs: list = range(1, 10),
                  cropping=False,
                  winLength=None,
                  step=1,
                  standardizing=True,
                  batch_size=10,
                  epochs=300,
                  patience=100,
                  verbose=2,
                  *args,
                  **kwargs):
        super().setConfig(built_fn=built_fn,
                          dataGent=dataGent,
                          splitMethod=splitMethod,
                          beg=beg,
                          end=end,
                          srate=srate,
                          kFold=kFold,
                          shuffle=shuffle,
                          random_state=random_state,
                          subs=subs,
                          cropping=cropping,
                          winLength=winLength,
                          step=step,
                          standardizing=standardizing,
                          batch_size=batch_size,
                          epochs=epochs,
                          patience=patience,
                          verbose=verbose,
                          *args,
                          **kwargs)
        _subs_targeted = False
        _subs_targeted_parameters = []
        for parameter in parameters:
            if not parameter in self.built_fn.__code__.co_varnames:
                raise ValueError('`parameters` has unsupported parameter in'
                                 ' `built_fn`.')
            if not isinstance(parameters[parameter], list) and not isinstance(
                    parameters[parameter], dict):
                parameters[parameter] = list(parameters[parameter])
            if isinstance(parameters[parameter], dict):
                subs = list(self.subs).copy()
                for subject in parameters[parameter]:
                    if not int(subject) in self.subs:
                        raise ValueError('`parameters` has unsolved subject'
                                         ' numbers.')
                    if not isinstance(parameters[parameter][subject], list):
                        parameters[parameter][subject] = list(
                            parameters[parameter][subject])
                    subs.remove(int(subject))
                if subs:
                    raise ValueError('`parameters` doesn\'t include all the'
                                     ' subject numbers.')
                _subs_targeted = True
                _subs_targeted_parameters.append(parameter)
        temp = []
        if _subs_targeted:
            for subject in self.subs:
                items = []
                for parameter in parameters:
                    if parameter in _subs_targeted_parameters:
                        items += list(
                            {parameter:
                             parameters[parameter][str(subject)]}.items())
                    else:
                        items += list({parameter:
                                       parameters[parameter]}.items())
                temp.append(dict(items))
        else:
            for subject in self.subs:
                temp.append(parameters)

        self.parameters = temp