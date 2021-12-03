# coding:utf-8
'''
Data loading and pre-processing methods are defined here.
'''
import os
import math
import numpy as np

from abc import ABCMeta, abstractmethod
from core.utils import load_data, highpassfilter, bandpassfilter
from core.utils import normalization, standardization


class BaseDataloader(object, metaclass=ABCMeta):
    '''
    Base class for all data loaders.
    
    Implementations must define `_load_data`.
    '''
    MODE_TRAIN = 'train'
    MODE_TEST = 'test'

    def __init__(self,
                 beg=0.,
                 end=4.,
                 srate=250,
                 traindata_filepath=None,
                 testdata_filepath=None,
                 datadir=None,
                 dn='',
                 norm_mode='maxmin',
                 name='BaseDataloader'):
        self.beg = beg
        self.end = end
        self.srate = srate
        self.traindata_filepath = traindata_filepath
        self.testdata_filepath = testdata_filepath
        self.datadir = datadir
        self.dn = dn
        self.norm_mode = norm_mode
        self.normalizing = norm_mode is not None
        self.name = name

        self.X1 = self.X2 = self.Y1 = self.Y2 = None
        self._readed = False

    def call(self):
        if not self._readed:
            self.X1, self.Y1 = self._read_data(subject=self.subject,
                                               mode=self.MODE_TRAIN)
            self.X2, self.Y2 = self._read_data(subject=self.subject,
                                               mode=self.MODE_TEST)
            if self.normalizing:
                self.X1, self.X2 = self._normalize((self.X1, self.X2),
                                                   mode=self.norm_mode)
            self._readed = True
        return self.X1, self.Y1, self.X2, self.Y2

    def __call__(self, subject):
        self.subject = subject
        return self.call()

    def _load_label(self, filepath):
        return load_data(filepath, label=True)

    @abstractmethod
    def _load_data(self, filepath):
        pass

    @staticmethod
    def _normalize(datas: tuple, trialaxis=0, mode='maxmin'):
        '''
        normalizing (maxmin) or standardizing (z-score) on each trial,
        supports np.nan numbers.
        '''
        if mode == 'maxmin':
            dofunc = normalization
        elif mode == 'z-score':
            dofunc = standardization
        else:
            raise ValueError('Parameter `norm_mode` wrong!')

        temp = tuple([])
        for data in datas:
            _len = len(data.shape)
            if _len > 1:
                axis = list(range(_len))
                axis.pop(trialaxis)
                axis = tuple(axis)
            else:
                axis = -1
            temp += (dofunc(data, axis=axis), )
        return temp

    def setReaded(self, readed: bool):
        self._readed = readed

    def _read_data(self, subject, mode):
        '''
        Read data from joined path.

        Parameters
        ----------
        ```txt
        subject : int, Identifier of subject.
        mode    : str, One of 'train' and 'test'.
        ```

        Return
        ------
        ```txt
        data    : tuple, (x, y).
        ```
        '''
        meta = [self.MODE_TEST, self.MODE_TRAIN]
        if not isinstance(mode, str):
            raise TypeError('`mode` must be passed as string.')
        if not mode in meta:
            raise ValueError('`mode` must be one of \'{}\' and \'{}\'.'.format(
                self.MODE_TEST, self.MODE_TRAIN))
        if mode == self.MODE_TEST:
            if not self.testdata_filepath:
                self.testdata_filepath = os.path.join(
                    self.datadir, 'TestSet',
                    self.dn + '0' + str(subject) + 'E.mat')
                x = self._load_data(self.testdata_filepath)
                y = self._load_label(self.testdata_filepath)
                self.testdata_filepath = None
                return x, y
            else:
                x = self._load_data(self.testdata_filepath)
                y = self._load_label(self.testdata_filepath)
                return x, y
        else:
            if not self.traindata_filepath:
                self.traindata_filepath = os.path.join(
                    self.datadir, 'TrainSet',
                    self.dn + '0' + str(subject) + 'T.mat')
                x = self._load_data(self.traindata_filepath)
                y = self._load_label(self.traindata_filepath)
                self.traindata_filepath = None
                return x, y
            else:
                x = self._load_data(self.traindata_filepath)
                y = self._load_label(self.traindata_filepath)
                return x, y


class RawDataloader(BaseDataloader):
    '''
    Raw data Generator.
    '''
    def __init__(self,
                 beg=0.,
                 end=4.,
                 srate=250,
                 traindata_filepath=None,
                 testdata_filepath=None,
                 datadir=None,
                 dn='',
                 norm_mode='maxmin',
                 name='RawDataloader'):
        super().__init__(beg=beg,
                         end=end,
                         srate=srate,
                         traindata_filepath=traindata_filepath,
                         testdata_filepath=testdata_filepath,
                         datadir=datadir,
                         dn=dn,
                         norm_mode=norm_mode,
                         name=name)

    def _load_data(self, filepath):
        data = load_data(filepath, label=False)
        data = bandpassfilter(data, srate=self.srate)
        data = data[:, :,
                    math.floor(self.beg * self.srate):math.ceil(self.end *
                                                                self.srate),
                    np.newaxis]
        return data


# TODO: Leave-One-Subject-Out needs pre-load all subjects' data.
class _BaseL1SODataloader(BaseDataloader):
    '''
    Leave-One-Subject-Out data base Dataloader.
    '''
    def __init__(self, beg=0., end=4., srate=250, name='BaseL1SODataloader'):
        super().__init__(beg=beg, end=end, srate=srate, name=name)

    def _load_data(self, filepath):
        raise NotImplementedError


class RawL1SODataloader(_BaseL1SODataloader):
    '''
    Leave-One-Subject-Out raw data Dataloader.
    '''
    def __init__(self, beg=0., end=4., srate=250, name='RawL1SODataloader'):
        super().__init__(beg=beg, end=end, srate=srate, name=name)

    def _load_data(self, filepath):
        data = load_data(filepath, label=False)
        data = bandpassfilter(data, srate=self.srate)
        data = data[:, :,
                    math.floor(self.beg * self.srate):math.ceil(self.end *
                                                                self.srate),
                    np.newaxis]
        return data