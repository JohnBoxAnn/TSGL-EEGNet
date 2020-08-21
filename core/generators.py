# coding:utf-8
'''
Data Generators suit for corresponding training method are defined here.
'''
import math
import numpy as np

from abc import ABCMeta, abstractmethod
from core.utils import load_data, load_or_gen_filterbank_data, load_or_gen_interestingband_data, load_or_generate_images, highpassfilter, bandpassfilter


class BaseGenerator(object, metaclass=ABCMeta):
    '''
    Base class for all data Generators.
    
    Implementations must define `__init__` and `_load_data`.
    '''
    @abstractmethod
    def __init__(self, beg=0., end=4., srate=250):
        self.beg = beg
        self.end = end
        self.srate = srate
        self.name = 'base'

    def __call__(self, filepath):
        return self._load_data(filepath), self._load_label(filepath)

    def _load_label(self, filepath):
        return load_data(filepath, label=True)

    @abstractmethod
    def _load_data(self, filepath):
        pass


class graphGenerator(BaseGenerator):
    '''
    Graph data Generator.
    '''
    def __init__(self,
                 H=6,
                 W=7,
                 beg=0.,
                 end=4.,
                 srate=250,
                 mode='raw',
                 averageImages=1):
        super().__init__(beg=beg, end=end, srate=srate)
        self.H = H
        self.W = W
        self.mode = mode
        self.averageImages = averageImages
        self.name = 'graph'

    def _load_data(self, filepath):
        return load_or_generate_images(filepath,
                                       beg=self.beg,
                                       end=self.end,
                                       srate=self.srate,
                                       mode=self.mode,
                                       averageImages=self.averageImages,
                                       H=self.H,
                                       W=self.W)


class rawGenerator(BaseGenerator):
    '''
    Raw data Generator.
    '''
    def __init__(self, beg=0., end=4., srate=250):
        super().__init__(beg=beg, end=end, srate=srate)
        self.name = 'raw'

    def _load_data(self, filepath):
        data = load_data(filepath, label=False)
        data = bandpassfilter(data, srate=self.srate)
        data = data[:, :,
                    round(self.beg * self.srate):round(self.end *
                                                       self.srate), np.newaxis]
        return data


# TODO: Leave-One-Subject-Out needs pre-load all subjects' data.
class _BaseL1SOGenerator(BaseGenerator):
    '''
    Leave-One-Subject-Out data base Generator.
    '''
    def __init__(self, beg=0., end=4., srate=250):
        super().__init__(beg=beg, end=end, srate=srate)
        self.name = 'base L1SO'

    def _load_data(self, filepath):
        raise NotImplementedError


class rawL1SOGenerator(_BaseL1SOGenerator):
    '''
    Leave-One-Subject-Out raw data Generator.
    '''
    def __init__(self, beg=0., end=4., srate=250):
        super().__init__(beg=beg, end=end, srate=srate)
        self.name = 'raw L1SO'

    def _load_data(self, filepath):
        data = load_data(filepath, label=False)
        data = bandpassfilter(data, srate=self.srate)
        data = data[:, :,
                    math.floor(self.beg *
                               self.srate):math.ceil(self.end *
                                                     self.srate), np.newaxis]
        return data
