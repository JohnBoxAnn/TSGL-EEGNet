# coding:utf-8
'''
Batch Data Generators suited for "cropped training method" are defined here.
'''
import math
import copy
import logging
import numpy as np

from numbers import Integral
from typing import Any, Dict, Tuple, Generator
from abc import ABCMeta, abstractmethod
from core.utils import check_random_state


class BaseGenerator(object, metaclass=ABCMeta):
    '''
    Base class for all data Generators.
    
    Implementations must rewrite `call` and `_yield_data`.
    '''
    def __init__(self,
                 batch_size: int = None,
                 epochs: int = None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 shuffle=True,
                 random_state=None,
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=None,
                 max_crop=None,
                 name='BaseGenerator'):
        if not batch_size:
            batch_size = 10
        self.batch_size = batch_size
        if not epochs:
            epochs = 1
        self.epochs = epochs
        self.beg = beg
        self.end = end
        self.srate = srate
        self.shuffle = shuffle
        self.random_state = check_random_state(random_state)
        self.cropping = cropping
        self.winLength = winLength
        self.cpt = cpt
        self.step = step
        self.max_crop = max_crop
        self.name = name
        self.Samples = math.ceil(self.end * self.srate - self.beg * self.srate)

        # cropped training
        if self.cropping:
            if self.winLength:
                if not isinstance(self.winLength, int):
                    raise TypeError('`winLength` should be passed as int.')
                if self.winLength > self.Samples:
                    raise ValueError(
                        '`winLength` should less than or equal (`end` - '
                        '`beg`) * `srate`.')
            if self.cpt and not self.winLength:
                if isinstance(self.cpt, (float, int)):
                    if self.cpt <= self.end - self.beg:
                        self.winLength = int(self.cpt * self.srate)
                    else:
                        raise ValueError(
                            '`cpt` should less than or equal `end` - `beg`.')
                else:
                    raise TypeError('`cpt` should be passed as int or float.')
            if not self.winLength:
                self.winLength = 2 * self.srate
            self.Samples -= self.winLength
            if self.step:
                if not isinstance(self.step, int):
                    raise TypeError('`step` should be passed as int.')
            else:
                self.step = 4
            # Samples = trialLength - winLengthps
            # so here the end of the list is Samples + 1
            self._crop_indices = np.arange(0, self.Samples + 1, self.step)
            if self.max_crop:
                if not isinstance(self.max_crop, int):
                    raise TypeError('`max_crop` should be passed as int.')
                self._crop_indices = self._crop_indices[:self.max_crop]
            self.pieces = len(self._crop_indices)
            print('cropping into {:d} pieces.'.format(self.pieces))
        else:
            self.winLength = self.Samples
            self._crop_indices = [0]
            self.pieces = 1

    @abstractmethod
    def call(self, x, y, **kwargs):
        return self._yield_data(x, y)

    def __call__(self, x, y, **kwargs):
        return self.call(x, y, **kwargs)

    def _yield_data(self, x, y) -> Generator[Tuple, None, None]:
        '''generate data'''
        raise NotImplementedError

    def _cropping_data(self, datas: tuple):
        for i in self._crop_indices:
            temp = tuple([])
            for data in datas:
                temp += (data[:, :, i:i + self.winLength, :], )
            yield temp


class RawGenerator(BaseGenerator):
    '''
    Raw data Generator.
    '''
    def __init__(self,
                 batch_size=None,
                 epochs=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 shuffle=True,
                 random_state=None,
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=None,
                 max_crop=None,
                 name='RawGenerator'):
        super().__init__(batch_size=batch_size,
                         epochs=epochs,
                         beg=beg,
                         end=end,
                         srate=srate,
                         random_state=random_state,
                         shuffle=shuffle,
                         cropping=cropping,
                         winLength=winLength,
                         cpt=cpt,
                         step=step,
                         max_crop=max_crop,
                         name=name)

    def call(self, x, y, **kwargs):
        return super().call(x, y, **kwargs)

    def _yield_data(self, x, y):
        '''
        Generate (data, label).

        Parameters
        ----------
        ```txt
        x   : ndarray, datas.
        y   : ndarray, labels.
        ```
        Yields
        ------
        ```txt
        tuple, (x, y).
        ```
        '''
        for _ in np.arange(self.epochs):
            if self.shuffle:
                temp = list(zip(x, y))
                self.random_state.shuffle(temp)
                x = np.array([temp[i][0] for i in np.arange(len(temp))])
                y = np.array([temp[i][1] for i in np.arange(len(temp))])
            for (cpd_x, ) in self._cropping_data((x, )):
                for i in np.arange(math.ceil(len(y) / self.batch_size),
                                   dtype=np.int32):
                    yield (cpd_x[i * self.batch_size:(i + 1) *
                                 self.batch_size, :, :, :],
                           y[i * self.batch_size:(i + 1) * self.batch_size, :])


class ClassDataGenerator(object, metaclass=ABCMeta):
    '''Generate class data, for type checking'''
    def __init__(self, batch_size, **kwargs):
        self.batch_size = batch_size

    def _make_class_data(self, x,
                         y) -> Tuple[Dict[Any, np.ndarray], np.ndarray, int]:
        data = {}
        labels = np.unique(y)
        for label in labels:
            data[label] = x[np.squeeze(y == label)]
        steps = math.ceil(
            max(len(a) for a in (data[l] for l in labels)) / self.batch_size)

        return data, labels, steps


class CBBRGenerator(RawGenerator, ClassDataGenerator):
    '''
    Class-Balenced Batch of Raw data Generator.

    The real batch size is the number of classes * `batch_size`.
    '''
    def __init__(self,
                 batch_size=None,
                 epochs=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 shuffle=True,
                 random_state=None,
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=None,
                 max_crop=None,
                 name='CBBRGenerator'):
        super().__init__(batch_size=batch_size,
                         epochs=epochs,
                         beg=beg,
                         end=end,
                         srate=srate,
                         shuffle=shuffle,
                         random_state=random_state,
                         cropping=cropping,
                         winLength=winLength,
                         cpt=cpt,
                         step=step,
                         max_crop=max_crop,
                         name=name)

    def _complete_data(self, data: dict, labels,
                       steps) -> Dict[Any, np.ndarray]:
        '''repeating some samples in the class with fewer samples'''
        for label in labels:
            length = len(data[label])
            res = (length // self.batch_size) * self.batch_size
            num_sup = steps * self.batch_size - length
            if num_sup > 0:
                temp = copy.deepcopy(data[label])
                if num_sup > res:
                    logging.warning(
                        '{:s}: The number of rest samples is less than'
                        ' the number of supplemental samples.'.format(
                            self.name))
                    logging.warning(
                        '{:s}: Repeating data for more times.'.format(
                            self.name))
                    if num_sup > length:
                        for _ in np.arange(num_sup // length):
                            temp = np.concatenate((temp, data[label]), axis=0)
                        res = num_sup - (num_sup // length) * length
                    else:
                        subs = np.arange(0, res, dtype=int)
                        self.random_state.shuffle(subs)
                        n_t = self.batch_size - length + res
                        subs = subs[:n_t]
                        temp = np.concatenate((temp, data[label][subs]),
                                              axis=0)
                        res = length
                        num_sup -= n_t
                subs = np.arange(0, res, dtype=int)
                self.random_state.shuffle(subs)
                subs = subs[:num_sup]
                data[label] = np.concatenate((temp, data[label][subs]), axis=0)
        return data

    def call(self, x, y, **kwargs):
        '''
        Make sure that every class has the same number of samples
        by repeating some samples in the class with fewer samples.
        Every epoch may have different samples.
        Then generate `x, y` balancedly. 
        '''
        for _ in np.arange(self.epochs):
            if self.shuffle:
                temp = list(zip(x, y))
                self.random_state.shuffle(temp)
                x = np.array([temp[i][0] for i in np.arange(len(temp))])
                y = np.array([temp[i][1] for i in np.arange(len(temp))])
            data, labels, steps = self._make_class_data(x, y)
            data = self._complete_data(data, labels, steps)
            gen = {}
            for label in labels:
                gen[label] = self._yield_data(
                    data[label],
                    np.array([label] * (self.batch_size * steps))[:,
                                                                  np.newaxis])
            for _ in np.arange(steps * self.pieces):
                gen_data = []
                for label in labels:
                    # will not raise StopIteration
                    gen_data.append(next(gen[label]))
                yield (np.concatenate([t[0] for t in gen_data], axis=0),
                       np.concatenate([t[1] for t in gen_data], axis=0))

    def _yield_data(self, x, y):
        '''
        Generate (data, label) from dataGent.

        Parameters
        ----------
        ```txt
        x   : ndarray, datas.
        y   : ndarray, labels.
        ```
        Yields
        ------
        ```txt
        tuple, (x, y).
        ```
        '''
        for (cpd_x, ) in self._cropping_data((x, )):
            for i in np.arange(math.ceil(len(y) / self.batch_size),
                               dtype=np.int32):
                yield (cpd_x[i * self.batch_size:(i + 1) *
                             self.batch_size, :, :, :],
                       y[i * self.batch_size:(i + 1) * self.batch_size, :])


class EmbdGenerator(ClassDataGenerator):
    '''Generate embeddings, for type checking'''
    MODE_SA_EMBD = 'sa_embd'

    def _sa_Embeddings(self, x, y):
        data, labels, _ = self._make_class_data(x, y)
        embeddings = []
        for label in labels:
            embeddings.append(np.mean(data[label], axis=0))
        return np.array(embeddings)  # nctf


class RawandEmbdGenerator(RawGenerator, EmbdGenerator):
    '''
    A Raw data Generator.

    Using `mode = RawandEmbdGenerator.MODE_OA_EMBD`
    to return a superimposed average embedding.
    '''
    def __init__(self,
                 batch_size=None,
                 epochs=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 shuffle=True,
                 random_state=None,
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=None,
                 max_crop=None,
                 name='RawandEmbdGenerator'):
        super().__init__(batch_size=batch_size,
                         epochs=epochs,
                         beg=beg,
                         end=end,
                         srate=srate,
                         shuffle=shuffle,
                         random_state=random_state,
                         cropping=cropping,
                         winLength=winLength,
                         cpt=cpt,
                         step=step,
                         max_crop=max_crop,
                         name=name)

    def call(self, x, y, **kwargs):
        mode = None
        if 'mode' in kwargs:
            mode = kwargs['mode']
        if mode == self.MODE_SA_EMBD:
            embd = self._sa_Embeddings(x, y)
            return embd[:, :, :self.winLength, :]
        else:
            return super().call(x, y, **kwargs)


class CBBRandEmbdGenerator(CBBRGenerator, EmbdGenerator):
    '''
    A Class-Balenced Batch of Raw data Generator.

    Using `mode = CBBRandEmbdGenerator.MODE_OA_EMBD`
    to return a superimposed average embedding.
    '''
    def __init__(self,
                 batch_size=None,
                 epochs=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 shuffle=True,
                 random_state=None,
                 cropping=False,
                 winLength=None,
                 cpt=None,
                 step=None,
                 max_crop=None,
                 name='CBBRandEmbdGenerator'):
        super().__init__(batch_size=batch_size,
                         epochs=epochs,
                         beg=beg,
                         end=end,
                         srate=srate,
                         shuffle=shuffle,
                         random_state=random_state,
                         cropping=cropping,
                         winLength=winLength,
                         cpt=cpt,
                         step=step,
                         max_crop=max_crop,
                         name=name)

    def call(self, x, y, **kwargs):
        '''
        Make sure that every class has the same number of samples
        by repeating some samples in the class with fewer samples.
        Every epoch may have different samples.
        Then generate `x, y` balancedly. 
        '''
        mode = None
        if 'mode' in kwargs:
            mode = kwargs['mode']
        if mode == self.MODE_SA_EMBD:
            embd = self._sa_Embeddings(x, y)
            return embd[:, :, :self.winLength, :]
        else:
            return super().call(x, y, **kwargs)


def get_steps(gen: BaseGenerator, x, y, batch_size=None):
    if not batch_size:
        batch_size = gen.batch_size
    if not isinstance(batch_size, Integral):
        raise TypeError('`batch_size` should be an Integral.')
    if isinstance(gen, CBBRGenerator):
        _, _, steps = gen._make_class_data(x, y)
        return steps * gen.pieces
    return math.ceil(len(y) / batch_size) * gen.pieces