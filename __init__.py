# coding:utf-8
"""
Model TSGLEEGNet

@author: Boxann John
@date: 11/07/2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .core import models
from .core import train
from .core import utils
from .core import regularizers
from .core import generators
from .core import splits
from .core import layers
from .core import callbacks
from .core import visualization
from .core import constraints

import BCIIV2a as example


__name__ = 'TSGLEEGNet'
__all__ = {
    'models', 'train', 'utils', 'regularizers', 'generators', 'splits',
    'layers', 'callbacks', 'visualization', 'example'
}

__version__ = '1.0'
