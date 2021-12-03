# coding:utf-8

import os
import gc
import sys
import time
import itertools
import numpy as np
from numpy.lib.arraysetops import isin
import tensorflow as tf
from typing import ClassVar, Callable
from tensorflow.python.keras.api._v2.keras.models import load_model
from tensorflow.python.keras.api._v2.keras import backend as K

from core.dataloaders import BaseDataloader as _BaseDataloader
from core.splits import _BaseCrossValidator
from core.generators import BaseGenerator as _BaseGenerator, get_steps, EmbdGenerator
from core.callbacks import StatModelCheckpoint, StatEarlyStopping
from core.models import get_custom_objects
from core.utils import check_random_state, walk_files
from core.initializers import EmbeddingInit

_console = sys.stdout


class crossValidate(object):
    '''
    Class for K-fold Cross Validation.

    This framework can collect `model`, `loss`, `acc` and `history` from each fold, and 
    save them into files. 
    Data spliting methods from sklearn.model_selection are supported. you can pass the 
    classes as `splitMethod`. 

    This class has implemented a magic method `__call__()` wrapping `call()`, for which
    it can be used like a Callable object.

    Parameters
    ----------
    ```txt
    built_fn        : Callable, Create Training model which need to cross-validate.
                      Please using string 'create_' at the begining of function name, 
                      like 'create_modelname'. 
                      Returns a `tf.keras.Model`.
    dataLoader      : ClassVar, Loader data from files. 
                      Returns (x_train, y_trian, x_test, y_test).
    dataGent        : ClassVar, Generate data for @built_fn, shapes (n_trails, ...). 
                      Yields (x, y).
                      More details see core.generators.
    splitMethod     : ClassVar, Support split methods from module sklearn.model_selection.
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
    norm_mode       : string, None, 'maxmin' or 'z-score'. Default = 'maxmin'.
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

    class dataLoader:
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

    class dataGenerator:
        def __init__(self, *a, beg=0, end=4, srate=250, **kw):
            ...

        def __call__(self, x, y, mode):
            x, y = self._process_data(x, y)
            mode = str(mode, encoding='utf-8')
            return self._yield_data(x, y, mode)
        
        def _yield_data(self, x, y, mode):
            ...
        ...
    ...
    avg_acc, avg_kappa = crossValidate(
        create_model, 
        dataLoader=dataLoader，
        dataGenerator=dataGenerator, 
        splitMethod=StratifiedKFold,
        beg=0,
        end=4,
        srate=250,
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
                 built_fn: Callable[..., tf.keras.Model] = None,
                 dataLoader: _BaseDataloader = None,
                 splitMethod: _BaseCrossValidator = None,
                 dataGent: _BaseGenerator = None,
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
                 max_crop=None,
                 norm_mode='maxmin',
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 preserve_initfile=False,
                 reinit=True,
                 *args,
                 **kwargs):
        self.built_fn = built_fn
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
        self.max_crop = max_crop
        self.norm_mode = norm_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.preserve_initfile = preserve_initfile
        self.reinit = reinit
        self.args = args
        self.kwargs = kwargs
        self._check_params()

        if self.datadir:
            for root, dirs, files in os.walk(self.datadir):
                if files:
                    self.dn = files[0][0]
                    break
        else:
            self.dn = ''

        if not isinstance(dataLoader, _BaseDataloader):
            self.dataLoader: _BaseDataloader = dataLoader(
                beg=self.beg,
                end=self.end,
                srate=self.srate,
                traindata_filepath=self.traindata_filepath,
                testdata_filepath=self.testdata_filepath,
                datadir=self.datadir,
                dn=self.dn,
                norm_mode=self.norm_mode)
        else:
            self.dataLoader = dataLoader

        if not isinstance(dataGent, _BaseGenerator):
            self.dataGent: _BaseGenerator = dataGent(
                batch_size=self.batch_size,
                epochs=self.epochs,
                beg=self.beg,
                end=self.end,
                srate=self.srate,
                cropping=self.cropping,
                winLength=self.winLength,
                cpt=self.cpt,
                step=self.step,
                max_crop=self.max_crop)
        else:
            self.dataGent = dataGent
        self.Samples = self.dataGent.winLength

        if built_fn:
            self.modelstr = built_fn.__name__[7:]
        if self.splitMethod:
            if self.splitMethod.__name__ == 'AllTrain':
                self.validation_name = 'Average Validation'
            else:
                self.validation_name = 'Cross Validation'

        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('result'):
            os.makedirs('result')

    def call(self, *args, **kwargs):
        initfile = os.path.join('.', 'CV_initweight.h5')
        tm = time.localtime()
        dirname = (
            'CV_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_{6:s}'.
            format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                   tm.tm_sec, self.modelstr))
        if not os.path.exists(os.path.join('model', dirname)):
            os.mkdir(os.path.join('model', dirname))
        if not os.path.exists(os.path.join('result', dirname)):
            os.mkdir(os.path.join('result', dirname))

        if not self.reinit:
            kwas = self._check_builtfn_params(**kwargs)
            model = self.built_fn(*args, **kwas, Samples=self.Samples)
            model.save(initfile)

        earlystopping = StatEarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=self.patience,
                                          verbose=0,
                                          mode='auto',
                                          statistic_best=True,
                                          p=0.05)

        avg_acci = []
        avg_kappai = []
        for self.subject in self.subs:
            k = 0
            accik = []
            kappaik = []
            rawdata = self.get_data()
            for data in self.get_split(rawdata):
                k += 1
                if self.reinit:
                    kwas = self._check_builtfn_params(**kwargs, data=data)
                    summary = False
                    if k == 1:
                        summary = True
                    model = self.built_fn(*args,
                                          **kwas,
                                          Samples=self.Samples,
                                          summary=summary)
                else:
                    model = load_model(initfile,
                                       custom_objects=get_custom_objects(
                                           self.modelstr))

                filename = ''
                for key in kwargs.keys():
                    if key in ['l1', 'l21', 'tl1']:
                        filename += '{0:s}({1:.8f})_'.format(key, kwargs[key])
                    elif isinstance(kwargs[key], int):
                        filename += '{0:s}({1:0>2d})_'.format(key, kwargs[key])

                filepath = os.path.join(
                    'result', dirname, filename +
                    '{:0>2d}_{:s}.txt'.format(self.subject, self.modelstr))
                with open(filepath, 'w+') as f:
                    sys.stdout = f
                    print(('{0:s} {1:d}-fold ' + self.validation_name +
                           ' Accuracy').format(self.modelstr, self.kFold))
                    print('Subject {:0>2d} fold {:0>2d} in processing'.format(
                        self.subject, k))
                    sys.stdout = _console
                    f.seek(0, 0)
                    for line in f.readlines():
                        print(line)
                    f.close()

                filepath = os.path.join(
                    'model', dirname,
                    filename + self.dn + '0{0:d}T_{1:s}({2:d}).h5'.format(
                        self.subject, self.modelstr, k))
                checkpointer = StatModelCheckpoint(filepath=filepath,
                                                   verbose=1,
                                                   save_best_only=True,
                                                   statistic_best=True,
                                                   p=0.05)

                # train model
                history = model.fit(
                    x=self.get_dataset(data['x_train'], data['y_train']),
                    epochs=self.epochs,
                    callbacks=[checkpointer, earlystopping],
                    verbose=self.verbose,
                    validation_data=self.get_dataset(data['x_val'],
                                                     data['y_val']),
                    steps_per_epoch=get_steps(self.dataGent, data['x_train'],
                                              data['y_train'],
                                              self.batch_size),
                    validation_steps=get_steps(self.dataGent, data['x_val'],
                                               data['y_val'], self.batch_size),
                ).history

                # load the best model to evaluate
                model.load_weights(filepath)

                # test model
                loss, acc, kappa = model.evaluate(
                    self.get_dataset(data['x_test'], data['y_test']),
                    verbose=self.verbose,
                    steps=get_steps(self.dataGent, data['x_test'],
                                    data['y_test'], self.batch_size),
                )

                # TODO: better prediction for cropped training
                # pred = model.predict(
                #     self.get_dataset(data['x_test'], data['y_test']),
                #     verbose=self.verbose,
                #     steps=math.ceil(len(data['y_test']) / self.batch_size)
                #     * self.dataGent.pieces,
                # )
                # pred = np.argmax(pred, axis=0)
                # pred = np.reshape(
                #     pred,
                #     (math.ceil(len(data['y_test']) / self.batch_size),
                #      self.dataGent.pieces))

                # save the train history
                filepath = filepath[:-3] + '.npy'
                np.save(filepath, history)

                # reset model's weights to train a new one next fold
                if os.path.exists(initfile) and not self.reinit:
                    model.reset_states()
                    model.load_weights(initfile)

                if self.reinit:
                    K.clear_session()
                    gc.collect()

                accik.append(acc)
                kappaik.append(kappa)
            avg_acci.append(np.average(np.array(accik)))
            avg_kappai.append(np.average(np.array(kappaik)))
            self.dataLoader.setReaded(False)
            del rawdata, data

            filename = ''
            for key in kwargs.keys():
                if key in ['l1', 'l21', 'tl1']:
                    filename += '{0:s}({1:.8f})_'.format(key, kwargs[key])
                elif isinstance(kwargs[key], int):
                    filename += '{0:s}({1:0>2d})_'.format(key, kwargs[key])

            filepath = os.path.join(
                'result', dirname, filename +
                '{:0>2d}_{:s}.txt'.format(self.subject, self.modelstr))
            with open(filepath, 'w+') as f:
                sys.stdout = f
                print(('{0:s} {1:d}-fold ' + self.validation_name +
                       ' Accuracy').format(self.modelstr, self.kFold))
                for ik in range(self.kFold):
                    print('Fold {:0>2d}: {:.2%} ({:.4f})'.format(
                        ik + 1, accik[ik], kappaik[ik]))
                print('Average   : {0:.2%} ({1:.4f})'.format(
                    avg_acci[-1], avg_kappai[-1]))
                sys.stdout = _console
                f.seek(0, 0)
                for line in f.readlines():
                    print(line)
                f.close()
        del model
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
            'dataLoader': self.dataLoader.__class__,
            'splitMethod': self.splitMethod.__class__,
            'dataGent': self.dataGent.__class__,
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
            'max_crop': self.max_crop,
            'norm_mode': self.norm_mode,
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
                  built_fn: Callable[..., tf.keras.Model] = None,
                  dataLoader: _BaseDataloader = None,
                  splitMethod: _BaseCrossValidator = None,
                  dataGent: _BaseGenerator = None,
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
                  max_crop=None,
                  norm_mode='maxmin',
                  batch_size=10,
                  epochs=300,
                  patience=100,
                  verbose=2,
                  preserve_initfile=False,
                  reinit=True,
                  *args,
                  **kwargs):
        self.built_fn = built_fn
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
        self.max_crop = max_crop
        self.norm_mode = norm_mode
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.verbose = verbose
        self.preserve_initfile = preserve_initfile
        self.reinit = reinit
        self.args = args
        self.kwargs = kwargs
        self._check_params()

        if self.datadir:
            for root, dirs, files in os.walk(self.datadir):
                if files:
                    self.dn = files[0][0]
                    break
        else:
            self.dn = ''

        if not isinstance(dataLoader, _BaseDataloader):
            self.dataLoader: _BaseDataloader = dataLoader(
                beg=self.beg,
                end=self.end,
                srate=self.srate,
                traindata_filepath=self.traindata_filepath,
                testdata_filepath=self.testdata_filepath,
                datadir=self.datadir,
                dn=self.dn,
                norm_mode=self.norm_mode)
        else:
            self.dataLoader = dataLoader

        if not isinstance(dataGent, _BaseGenerator):
            self.dataGent: _BaseGenerator = dataGent(
                batch_size=self.batch_size,
                epochs=self.epochs,
                beg=self.beg,
                end=self.end,
                srate=self.srate,
                cropping=self.cropping,
                winLength=self.winLength,
                cpt=self.cpt,
                step=self.step,
                max_crop=self.max_crop)
        else:
            self.dataGent = dataGent
        self.Samples = self.dataGent.winLength

        if built_fn:
            self.modelstr = built_fn.__name__[7:]
        if self.splitMethod:
            if self.splitMethod.__name__ == 'AllTrain':
                self.validation_name = 'Average Validation'
            else:
                self.validation_name = 'Cross Validation'

        if not os.path.exists('model'):
            os.makedirs('model')
        if not os.path.exists('result'):
            os.makedirs('result')

    def get_data(self):
        data = {
            'x_train': None,
            'y_train': None,
            'x_val': None,
            'y_val': None,
            'x_test': None,
            'y_test': None
        }
        data['x_train'], data['y_train'], data['x_test'], data[
            'y_test'] = self.dataLoader(self.subject)
        return data

    def get_split(self, data, **kwargs):
        groups = None
        if 'groups' in kwargs:
            groups = kwargs['groups']

        for (x1, y1), (x2, y2) in self._split(data['x_train'],
                                              data['y_train'],
                                              groups=groups):
            if x2 is None:
                x2, y2 = data['x_test'], data['y_test']
            data['x_train'] = x1
            data['y_train'] = y1
            data['x_val'] = x2
            data['y_val'] = y2
            yield data

    def get_generator(self, x, y):
        return self.dataGent(x, y)

    def get_dataset(self, x, y):
        dataset = tf.data.Dataset.from_generator(
            self.get_generator, (tf.float32, tf.int32),
            output_shapes=(tf.TensorShape([
                None,
            ] + [x.shape[1], self.Samples, x.shape[3]]),
                           tf.TensorShape([
                               None,
                           ] + list(y.shape[1:]))),
            args=(x, y))
        # dataset = dataset.repeat()
        # print(list(dataset.as_numpy_iterator())[0][0].shape)
        # print(len(list(dataset.as_numpy_iterator())))
        return dataset

    def _split(self, X, y, groups=None):
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
        train : tuple, (x, y)
            The training set for that split.

        val : tuple, (x, y)
            The validating set for that split.
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
        if not isinstance(self.built_fn, Callable):
            raise TypeError('`built_fn` should be passed as a callable.')
        if not self.built_fn.__name__.split('_')[0] == 'create':
            raise ValueError('`built_fn` should be named as "create_*".')
        self.random_state = check_random_state(self.random_state)
        self.subject = None

    def _check_builtfn_params(self, data=None, **kwargs):
        '''
        Cross Validate check `built_fn`'s parameters out.
        '''
        if self.modelstr in ('EEGTransformer_Learnembd',
                             'EEGTransformer_Batchselfembd'):
            if 'modeldir' in kwargs:
                modeldir = kwargs['modeldir']
                del kwargs['modeldir']
                modelpath = walk_files(
                    os.path.join(modeldir, '{:0>2d}'.format(self.subject)),
                    'h5')[0]
            elif 'modelpath' in kwargs:
                modelpath = kwargs['modelpath']
                del kwargs['modelpath']
            else:
                raise ValueError('{} should have parameter `modeldir` or '
                                 '`modelpath` passed in.'.format(
                                     self.modelstr))
            model: tf.keras.Model = load_model(
                modelpath,
                custom_objects=get_custom_objects('EEGTransformer_Batchembd'),
                compile=False)
            _input = model.layers[0].input
            _output = model.layers[-2].output
            model = tf.keras.Model(_input, _output)
            for layer in model.layers:
                layer.trainable = False
            kwargs.update({'model': model})
            del model
        if self.modelstr in ('EEGTransformer_Learnembd'):
            if not isinstance(self.dataGent, EmbdGenerator):
                raise TypeError('`{}` should use a kind of `EmbdGenerator`'
                                ', not `{}`.'.format(
                                    self.modelstr,
                                    type(self.dataGent).__name__))
            if not isinstance(data, dict):
                raise TypeError('`EEGTransformer_Learnembd` needs `data` '
                                'passed as a dict.')
            _input = kwargs['model'].layers[0].input
            _output = kwargs['model'].layers[3].output
            model = tf.keras.Model(_input, _output)
            embdinit = model(
                self.dataGent(data['x_train'],
                              data['y_train'],
                              mode=self.dataGent.MODE_SA_EMBD)).numpy()
            print(type(embdinit))
            embdinit = EmbeddingInit(embdinit)
            kwargs.update({'embdinit': embdinit})
            del model
        for key in kwargs:
            if not key in self.built_fn.__code__.co_varnames:
                raise ValueError(
                    '`{}` is unsupported parameter in `{}`.'.format(
                        key, self.built_fn.__name__))
        return kwargs


class gridSearch(crossValidate):
    '''
    Class for K-fold Cross Validation Grid Search.

    Grid Search method. A subclass of `crossValidate`. 

    This framework can collect `model`, `loss`, `acc` and `history` from each fold, and 
    save them into files. 
    Data spliting methods from sklearn.model_selection are supported. you can pass the 
    classes as `splitMethod`. 

    It can't use multiple GPUs to speed up now. To grid search on a large parameter 
    matrix, you should use Greedy Algorithm.

    This class has implemented a magic method `__call__()` wrapping `call()`, for which
    it can be used like a Callable object.

    Parameters
    ----------
    ```txt
    built_fn        : Callable, Create Training model which need to cross-validate.
                      Please using string 'create_' at the begining of function name, 
                      like 'create_modelname'.
    parameters      : dict, Parameters need to grid-search. Keys are the parameters' 
                      name, and every parameter values are vectors which should be 
                      passed as a list.
    dataLoader      : ClassVar, Loader data from files. 
                      Returns (x_train, y_trian, x_test, y_test).
    dataGent        : ClassVar, Generate data for @built_fn, shapes (n_trails, ...). 
                      Yields (x, y).
                      More details see core.generators.
    splitMethod     : ClassVar, Support split methods from module sklearn.model_selection.
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
    norm_mode       : string, None, 'maxmin` or 'z-score'. Default = 'maxmin'.
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

    class dataLoader:
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

    class dataGenerator:
        def __init__(self, *a, beg=0, end=4, srate=250, **kw):
            ...

        def __call__(self, x, y, mode):
            x, y = self._process_data(x, y)
            mode = str(mode, encoding='utf-8')
            return self._yield_data(x, y, mode)
        
        def _yield_data(self, x, y, mode):
            ...
        ...
    ...
    parameters = {'para1':[...], 'para2':[...], ...}
    avg_acc, avg_kappa = gridSearch(
        create_model, 
        parameters,
        dataLoader=dataLoader,
        dataGenerator=dataGenerator, 
        splitMethod=StratifiedKFold,
        beg=0,
        end=4,
        srate=250,
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
                 built_fn: Callable[..., tf.keras.Model] = None,
                 parameters: dict = {},
                 dataLoader: _BaseDataloader = None,
                 dataGent: _BaseGenerator = None,
                 splitMethod: _BaseCrossValidator = None,
                 traindata_filepath=None,
                 testdata_filepath=None,
                 datadir=None,
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
                 max_crop=None,
                 norm_mode='maxmin',
                 batch_size=10,
                 epochs=300,
                 patience=100,
                 verbose=2,
                 preserve_initfile=False,
                 reinit=False,
                 *args,
                 **kwargs):
        super().__init__(built_fn=built_fn,
                         dataLoader=dataLoader,
                         dataGent=dataGent,
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
                         max_crop=max_crop,
                         norm_mode=norm_mode,
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
            for subject in range(1, max(self.subs) + 1):
                items = []
                for parameter in parameters:
                    if subject in self.subs:
                        if parameter in _subs_targeted_parameters:
                            items += list({
                                parameter:
                                parameters[parameter][str(subject)]
                            }.items())
                        else:
                            items += list({parameter:
                                           parameters[parameter]}.items())
                temp.append(dict(items))
        else:
            for subject in range(1, max(self.subs) + 1):
                if subject in self.subs:
                    temp.append(parameters)
                else:
                    temp.append([])

        self.parameters = temp

    def call(self, *args, **kwargs):
        '''
        parameters should be lists to different subjects, then pass one 
        subject's parameter to cv.
        '''
        initfile = os.path.join('.', 'GSCV_initweight.h5')
        tm = time.localtime()
        dirname = (
            'GS_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_{6:s}'.
            format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour, tm.tm_min,
                   tm.tm_sec, self.modelstr))
        if not os.path.exists(os.path.join('model', dirname)):
            os.mkdir(os.path.join('model', dirname))
        if not os.path.exists(os.path.join('result', dirname)):
            os.mkdir(os.path.join('result', dirname))

        earlystopping = StatEarlyStopping(monitor='val_loss',
                                          min_delta=0,
                                          patience=self.patience,
                                          verbose=0,
                                          mode='auto',
                                          statistic_best=True,
                                          p=0.05)

        def cv(*args, **kwargs):
            '''one subject, one parameter'''
            kwas = self._check_builtfn_params(**kwargs)
            if not self.reinit:
                if not os.path.exists(initfile):
                    model = self.built_fn(*args, **kwas, Samples=self.Samples)
                    model.save_weights(initfile)
                else:
                    model = self.built_fn(*args, **kwas, Samples=self.Samples)
                    model.load_weights(initfile)

            filename = ''
            for key in kwargs.keys():
                if key in ['l1', 'l21', 'tl1']:
                    filename += '{0:s}({1:.8f})_'.format(key, kwargs[key])
                elif isinstance(kwargs[key], int):
                    filename += '{0:s}({1:0>2d})_'.format(key, kwargs[key])

            k = 0
            acck = []
            kappak = []
            rawdata = self.get_data()
            for data in self.get_split(rawdata):
                k += 1
                if self.reinit:
                    summary = False
                    if k == 1:
                        summary = True
                    kwas = self._check_builtfn_params(**kwargs, data=data)
                    model = self.built_fn(*args,
                                          **kwas,
                                          Samples=self.Samples,
                                          summary=summary)
                else:
                    model.load_weights(initfile)

                filepath = os.path.join(
                    'model', dirname,
                    filename + self.dn + '0{0:d}T_{1:s}({2:d}).h5'.format(
                        self.subject, self.modelstr, k))
                checkpointer = StatModelCheckpoint(filepath=filepath,
                                                   verbose=1,
                                                   save_best_only=True,
                                                   statistic_best=True,
                                                   p=0.05)

                # train model
                history = model.fit(
                    x=self.get_dataset(data['x_train'], data['y_train']),
                    epochs=self.epochs,
                    callbacks=[checkpointer, earlystopping],
                    verbose=self.verbose,
                    validation_data=self.get_dataset(data['x_val'],
                                                     data['y_val']),
                    steps_per_epoch=get_steps(self.dataGent, data['x_train'],
                                              data['y_train'],
                                              self.batch_size),
                    validation_steps=get_steps(self.dataGent, data['x_val'],
                                               data['y_val'], self.batch_size),
                ).history

                # load the best model to evaluate
                model.load_weights(filepath)

                # test model
                loss, acc, kappa = model.evaluate(
                    self.get_dataset(data['x_test'], data['y_test']),
                    verbose=self.verbose,
                    steps=get_steps(self.dataGent, data['x_test'],
                                    data['y_test'], self.batch_size),
                )

                # save the train history
                npy_filepath = filepath[:-3] + '.npy'
                np.save(npy_filepath, history)

                # reset model's weights to train a new one next fold
                if os.path.exists(initfile) and not self.reinit:
                    model.reset_states()
                    model.load_weights(initfile)

                if self.reinit:
                    K.clear_session()
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
                'result', dirname, filename + self.dn +
                '0{0:d}T_{1:s}.txt'.format(self.subject, self.modelstr))
            with open(filepath, 'w+') as f:
                sys.stdout = f
                print(('{0:s} {1:d}-fold ' + self.validation_name +
                       ' Accuracy').format(self.modelstr, self.kFold))
                print('Subject {0:0>2d}'.format(self.subject))
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
        filepath = os.path.join(
            'result',
            'GS_{0:d}_{1:0>2d}_{2:0>2d}_{3:0>2d}_{4:0>2d}_{5:0>2d}_' \
            '{6:s}.txt'.format(tm.tm_year, tm.tm_mon, tm.tm_mday, tm.tm_hour,
                               tm.tm_min, tm.tm_sec, self.modelstr))
        for self.subject in self.subs:
            parameters.append(self._combination(subject=self.subject))
            count = 0
            with open(filepath, 'w+') as f:
                sys.stdout = f
                print('Subject: {0:0>2d}/{1:0>2d}'.format(
                    self.subject, len(self.subs)))
                print(
                    'Grid Search progress: {0:0>4d}/{1:0>4d}\n' \
                    'The No.{2:0>4d} is in processing'
                    .format(count, len(parameters[-1]), count + 1))
                sys.stdout = _console
                f.seek(0, 0)
                for line in f.readlines():
                    print(line)
                f.close()
            avg_acc = []
            avg_kappa = []
            for parameter in parameters[-1]:
                param = dict(parameter + list(kwargs.items()))
                acc, kappa = cv(*args, **param)
                avg_acc.append(acc)
                avg_kappa.append(kappa)
                count += 1
                with open(filepath, 'w+') as f:
                    sys.stdout = f
                    print('Subject: {0:0>2d}/{1:0>2d}'.format(
                        self.subject, len(self.subs)))
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
            self.dataLoader.setReaded(False)
            max_avg_acc.append(np.max(avg_acc))
            indices.append(np.argmax(avg_acc))
            max_acc_kappa.append(avg_kappa[indices[-1]])
        if os.path.exists(initfile) and not self.preserve_initfile:
            os.remove(initfile)

        with open(filepath, 'w+') as f:
            sys.stdout = f
            print(('{0:s} {1:d}-fold ' + self.validation_name +
                   'Grid Search Accuracy (kappa)').format(
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
                  built_fn: Callable[..., tf.keras.Model] = None,
                  parameters: dict = {},
                  dataLoader: _BaseDataloader = None,
                  dataGent: _BaseGenerator = None,
                  splitMethod: _BaseCrossValidator = None,
                  traindata_filepath=None,
                  testdata_filepath=None,
                  datadir=None,
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
                  norm_mode='maxmin',
                  batch_size=10,
                  epochs=300,
                  patience=100,
                  verbose=2,
                  preserve_initfile=False,
                  reinit=False,
                  *args,
                  **kwargs):
        super().setConfig(built_fn=built_fn,
                          dataLoader=dataLoader,
                          dataGent=dataGent,
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
            for subject in range(1, max(self.subs) + 1):
                items = []
                for parameter in parameters:
                    if subject in self.subs:
                        if parameter in _subs_targeted_parameters:
                            items += list({
                                parameter:
                                parameters[parameter][str(subject)]
                            }.items())
                        else:
                            items += list({parameter:
                                           parameters[parameter]}.items())
                temp.append(dict(items))
        else:
            for subject in range(1, max(self.subs) + 1):
                if subject in self.subs:
                    temp.append(parameters)
                else:
                    temp.append([])

        self.parameters = temp