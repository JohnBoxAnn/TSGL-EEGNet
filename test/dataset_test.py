#coding:utf-8
import __init__
import os
import numpy as np
import tensorflow as tf
from typing import ClassVar

from core.generators import RawGenerator
from core.generators import BaseGenerator as _BaseGenerator
from core.splits import StratifiedKFold, _BaseCrossValidator


class DatasetTest():
    def __init__(self,
                 dataGent: _BaseGenerator,
                 splitMethod: _BaseCrossValidator = StratifiedKFold,
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
                 *args,
                 **kwargs):
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
        self.args = args
        self.kwargs = kwargs
        self.Samples = np.math.ceil(self.end * self.srate -
                                    self.beg * self.srate)

        if self.datadir:
            for root, dirs, files in os.walk(self.datadir):
                if files:
                    self.dn = files[0][0]
                    break
        else:
            self.dn = ''

        self.dataGent: _BaseGenerator = dataGent(
            batch_size=batch_size,
            beg=beg,
            end=end,
            srate=srate,
            splitMethod=splitMethod,
            traindata_filepath=traindata_filepath,
            testdata_filepath=testdata_filepath,
            datadir=datadir,
            dn=self.dn,
            kFold=kFold,
            shuffle=shuffle,
            random_state=random_state,
            cropping=cropping,
            winLength=winLength,
            cpt=cpt,
            step=step,
            max_crop=max_crop,
            norm_mode=norm_mode,
            *args,
            **kwargs)

    def call(self):
        pass

    def __call__(self):
        return self.call()

    def _get_dataset(self, mode: str):
        '''
        Return a dataset that yields (data, label) from the dataGent
        with explicit `mode`. `modes` are defined in the dataGent.

        Parameters
        ----------
        ```txt
        mode        : str, defined in the dataGent.
        ```

        Return
        ------
        ```txt
        dataset     : tf.data.Dataset. Yields (data, label).
        ```
        '''
        if mode == self.dataGent.MODE_TRAIN:
            return tf.data.Dataset.from_generator(self._gent_train_data,
                                                  (tf.float32, tf.float32))
        if mode == self.dataGent.MODE_VAL:
            return tf.data.Dataset.from_generator(self._gent_val_data,
                                                  (tf.float32, tf.float32))
        if mode == self.dataGent.MODE_TEST:
            return tf.data.Dataset.from_generator(self._gent_test_data,
                                                  (tf.float32, tf.float32))
        raise ValueError('`mode` is not defined in the dataGent.')

    def _gent_train_data(self):
        '''
        Generate training data (data, label) from dataGent.

        Parameters
        ----------
        ```txt
        subject     : int, Identifier of subject.
        ```

        Yields
        ------
        ```txt
        data        : tuple, (x, y).
        ```
        '''
        for x, y in self.dataGent(self.subject, self.dataGent.MODE_TRAIN):
            yield x, y

    def _gent_val_data(self):
        '''
        Generate validation data (data, label) from dataGent.

        Parameters
        ----------
        ```txt
        subject     : int, Identifier of subject.
        ```

        Yields
        ------
        ```txt
        data        : tuple, (x, y).
        ```
        '''
        for x, y in self.dataGent(self.subject, self.dataGent.MODE_VAL):
            yield x, y

    def _gent_test_data(self):
        '''
        Generate testing data (data, label) from dataGent.

        Parameters
        ----------
        ```txt
        subject     : int, Identifier of subject.
        ```

        Yields
        ------
        ```txt
        data        : tuple, (x, y).
        ```
        '''
        for x, y in self.dataGent(self.subject, self.dataGent.MODE_TEST):
            yield x, y


datadir = os.path.join('data', 'A')
kFold = 5
