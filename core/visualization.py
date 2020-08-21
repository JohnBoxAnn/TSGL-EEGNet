# coding:utf-8
import os
import pywt
import copy
import numpy as np
import tensorflow as tf
import scipy.signal as signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mne.viz as viz

from tensorflow.python.keras.api._v2.keras import backend as K
from tensorflow.python.keras.api._v2.keras import utils
from scipy.fftpack import fft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from core.generators import rawGenerator
from core.utils import interestingband, normalization
from core.utils import load_locs, azim_proj


# TODO: cropped visualizing
class visualize(object):
    '''
    Visualize Keras model from selected layers and data.

    Layers' name
    - tfconv    : time-frequency feather conv
    - sconv     : spatial feather conv
    - fs        : feather selection
    - flatten   : flatten
    - softmax   : output

    Useage
    ------
    >>> vis = visualize(model, vis_data)
    >>> vis.fft_output(layer_name)
    >>> vis.kernel(layer_name)
    >>> vis.show()
    '''
    def __init__(self,
                 model: tf.keras.Model,
                 vis_data=None,
                 vis_data_file=None,
                 cropping=False,
                 winLength=None,
                 step=None,
                 beg=0.,
                 end=4.,
                 srate=250,
                 dataGent=rawGenerator,
                 locpath=None):
        self.model = model
        self.layer_dict = dict([(layer.name, layer)
                                for layer in model.layers[1:]])
        self.data = vis_data
        if self.data is not None:
            if not isinstance(self.data, dict):
                raise TypeError('`vis_data` should be a dict')
            else:
                for key in self.data:
                    if key in ['x', 'y']:
                        if isinstance(self.data[key], list):
                            self.data[key] = np.array(self.data[key])
                        elif not isinstance(self.data[key], np.ndarray):
                            raise TypeError(
                                'data should be passed as list or ndarray')
                    else:
                        raise ValueError(
                            '`vis_data` should have keys \'x\' and \'y\'')
        self.data_file = vis_data_file
        if self.data_file and not isinstance(self.data_file, str):
            raise TypeError('`vis_data_file` should be passed as string')
        self.cropping = cropping
        self.winLength = winLength
        if not self.winLength:
            self.winLength = srate * 2
        self.step = step  # for future release
        if not self.step:
            self.step = 4
        self.dataGent = dataGent(beg=beg, end=end, srate=srate)
        if self.data is None:
            if self.data_file:
                self.data = self._read_data(srate)
            else:
                raise ValueError('no given data to compute layer outputs')
        if not locpath:
            locpath = os.path.join('.', 'data', '22scan_locs.mat')
        elif not isinstance(locpath, str):
            raise TypeError('`locpath` should be passed as string')
        locs_3d, names = load_locs(locpath)
        self.sensors_name = names
        self.locs = []
        # Convert to 2D
        for e in locs_3d:
            self.locs.append(azim_proj(e))
        self.locs = np.array(self.locs)

        self.class_names = {
            0: 'left hand',
            1: 'right hand',
            2: 'foot',
            3: 'tongue'
        }

        plt.rcParams['font.family'] = 'Times New Roman'

    def _read_data(self, srate):
        x, y = self.dataGent._load_data(self.data_file)
        if self.cropping:
            return {
                'x': x[:, :, 0.5 * srate:0.5 * srate + self.winLength, :],
                'y': y
            }
        else:
            return {'x': x, 'y': y}

    @staticmethod
    def _class_data(data):
        unique_y = np.unique(data['y'])
        cls_dict = {}
        for c in unique_y:
            indices = data['y'] == c
            cls_dict[int(c)] = {}
            cls_dict[int(c)]['x'] = data['x'][np.squeeze(indices)]
            cls_dict[int(c)]['y'] = c
        return cls_dict

    @staticmethod
    def _ban_axis(ax):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    def _check_layer_name(self, layer_name):
        if not isinstance(layer_name, list) and not isinstance(
                layer_name, tuple):
            if isinstance(layer_name, str):
                layer_name = [layer_name]
            else:
                raise ValueError(
                    '`layer_name` should be a string or a list of string')
        for name in layer_name:
            if not isinstance(name, str):
                raise TypeError(
                    '`layer_name` should be a string or a list of string')
            if not name in self.layer_dict.keys():
                raise ValueError(
                    'No layer named {} given in `layer_name`'.format(name))
        return layer_name

    def show(self):
        plt.show()  # block the thread

    def fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.layers[0].input
        plt.rcParams['font.size'] = 9
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            r = t(self.data['x'])  # ndarray
            a = []
            fig, axs = plt.subplots(_output.shape[-1] // 2, 2, sharex=True)
            for i in np.arange(_output.shape[-1]):
                fred = np.abs(fft(r[:, :, :, i]))
                fred = np.average(fred, axis=0)
                fred = fred / len(fred.T)
                fred = fred[:, :101]
                a.append(fred)
                axs[i // 2, i % 2].set_prop_cycle('color', [
                    plt.cm.Spectral_r(i) for i in np.linspace(0, 1, len(fred))
                ])
                for col in np.arange(len(fred)):
                    axs[i // 2, i % 2].plot(np.arange(len(fred.T)), fred[col])
                axs[i // 2, i % 2].autoscale(enable=True,
                                             axis='both',
                                             tight=True)
                axs[i // 2, i % 2].set_xlabel(chr(ord('a') + i))
            plt.subplots_adjust(right=0.98,
                                left=0.05,
                                top=0.99,
                                bottom=0.09,
                                wspace=0.15,
                                hspace=0.5)
            plt.show(block=False)
            a = np.average(np.array(a), axis=(0, 1))
            plt.figure()
            plt.plot(np.arange(len(a)), a)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.tight_layout()
            plt.show(block=False)

    def output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.layers[0].input
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            r = t(self.data['x'][0:1])  # ndarray
            plt.figure()
            for i in range(_output.shape[-1]):
                rci = np.squeeze(r[:, :, :, i], axis=0)
                plt.subplot(_output.shape[-1], 1, i + 1)
                plt.plot(np.arange(len(rci.T)), rci.T)
                plt.autoscale(enable=True, axis='both', tight=True)
                plt.tight_layout()
            plt.tight_layout()
            plt.show(block=False)

    def topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        plt.rcParams['font.size'] = 9
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            _min = np.min(_weights)
            _max = np.max(_weights)
            n = 1
            fig, axs = plt.subplots()
            for i in range(_weights.shape[-2]):
                for j in range(_weights.shape[-1]):
                    ax = plt.subplot(
                        _weights.shape[-2] * _weights.shape[-1] // 20, 20, n)
                    im, cn = viz.plot_topomap(
                        np.squeeze(_weights[:, :, i, j]),
                        self.locs,
                        vmin=_min,
                        vmax=_max,
                        names=self.sensors_name,
                        show_names=True,
                        show=False,
                        image_interp='bicubic',
                        extrapolate='head',
                        sphere=(0, 0, 0, 1))  # draw topographic image
                    if n <= 20:
                        ax.spines['bottom'].set_position(('axes', 1.2))
                        if n <= 10:
                            ax.set_xlabel(n, fontsize=12)
                        else:
                            ax.set_xlabel(n - 10, fontsize=12)
                    if n % 20 == 1:
                        ax.spines['left'].set_position(('axes', 0.07))
                        ax.set_ylabel(chr(ord('a') + n // 10), fontsize=12)
                    if n % 20 == 0:
                        ax.spines['left'].set_position(('axes', 1.22))
                        ax.set_ylabel(chr(ord('a') - 1 + n // 10), fontsize=12)
                    n += 1
            divider = make_axes_locatable(axs)
            cax = divider.append_axes('bottom', size='1%', pad=0.6)
            cb = fig.colorbar(im,
                              cax=cax,
                              orientation='horizontal',
                              ticklocation='top')
            plt.tight_layout(pad=0)
            plt.subplots_adjust(right=0.99,
                                left=0.01,
                                top=1,
                                bottom=0,
                                wspace=0,
                                hspace=0)
            plt.show(block=False)

    def fft_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            rci = np.squeeze(_weights)
            fred = np.abs(fft(rci.T))
            fred = fred / len(fred.T)
            fred = fred[:, :int((len(fred.T) / 2)) + 1]
            plt.figure()
            plt.title('Weights in layer {} after FFT'.format(name))
            plt.plot(np.arange(len(fred.T)), fred.T)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.tight_layout()
            plt.show(block=False)

    def kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        plt.rcParams['font.size'] = 16
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            fig, axs = plt.subplots()
            axs.set_title('Weights in layer {}'.format(name))
            axs.imshow(np.squeeze(_weights.T))
            plt.subplots_adjust(right=0.99,
                                left=0.06,
                                top=0.93,
                                bottom=0.07,
                                wspace=0,
                                hspace=0)
            plt.show(block=False)

    def depthwise_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        plt.rcParams['font.size'] = 16
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            fig, axs = plt.subplots()
            axs.set_title('Weights in layer {}'.format(name))
            axs.imshow(
                np.reshape(np.squeeze(_weights),
                           (_weights.shape[0],
                            _weights.shape[-1] * _weights.shape[-2])))
            plt.subplots_adjust(right=0.99,
                                left=0.06,
                                top=0.93,
                                bottom=0.07,
                                wspace=0,
                                hspace=0)
            plt.show(block=False)

    # TODO: fs visualiztion
    def fs_topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)

    def fs_fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.input
        _output = self.model.output
        for name in layer_name:
            layer = self.layer_dict[name]
            fig, axs = plt.subplots(layer.output.shape[-1] // 2,
                                    2,
                                    sharex=True)
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            with tf.GradientTape() as g:
                conv_output, Pred = layer_model(self.data['x'])
                index = np.argmax(Pred[0])
                prob = Pred[:, index]
                grads = g.gradient(prob, conv_output)
                pooled_grads = K.mean(grads, axis=(0, 1, 2))
            selected = tf.reduce_mean(tf.multiply(pooled_grads, conv_output),
                                      axis=0)
            for i in np.arange(selected.shape[-1]):
                fred = np.abs(fft(np.array(selected[:, :, i]), axis=1))
                fred = fred / len(fred.T)
                fred = fred[:, :101]
                axs[i // 2, i % 2].set_prop_cycle('color', [
                    plt.cm.Spectral_r(i) for i in np.linspace(0, 1, len(fred))
                ])
                for col in np.arange(len(fred)):
                    axs[i // 2, i % 2].plot(np.arange(len(fred.T)), fred[col])
                axs[i // 2, i % 2].autoscale(enable=True,
                                             axis='both',
                                             tight=True)
                axs[i // 2, i % 2].set_xlabel(chr(ord('a') + i))
            plt.subplots_adjust(right=0.98,
                                left=0.05,
                                top=0.99,
                                bottom=0.09,
                                wspace=0.15,
                                hspace=0.5)
            plt.show(block=False)
            a = np.average((np.abs(fft(np.array(selected), axis=1)) /
                            selected.shape[1])[:, :101, :],
                           axis=(0, -1))
            plt.figure()
            plt.plot(np.arange(len(a)), a)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.tight_layout()
            plt.show(block=False)

    # TODO: class visualization
    def class_fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.layers[0].input
        plt.rcParams['font.size'] = 12
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            fig, axs = plt.subplots(2, 2, sharex=True)
            for c in class_data:
                r = t(class_data[c]['x'])  # ndarray
                fred = np.abs(fft(r, axis=2))
                fred = np.average(fred, axis=(0, 1, -1))
                fred = fred / len(fred)
                fred = fred[:101]
                axs[c // 2, c % 2].plot(np.arange(len(fred)), fred)
                axs[c // 2,
                    c % 2].set_xlabel('({}) {}'.format(chr(c + 97),
                                                       self.class_names[c]))
                axs[c // 2, c % 2].autoscale(enable=True,
                                             axis='both',
                                             tight=True)
            plt.subplots_adjust(right=0.97,
                                left=0.07,
                                top=0.99,
                                bottom=0.10,
                                wspace=0.17,
                                hspace=0.10)
            plt.show(block=False)

    def fs_class_topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.input
        _output = self.model.output
        plt.rcParams['font.size'] = 12
        for name in layer_name:
            layer = self.layer_dict[name]
            _weights = layer.get_weights()[0]
            layer_model = tf.keras.Model([_input],
                                         [layer.input, layer.output, _output])
            fig = plt.figure()
            for c in class_data:
                with tf.GradientTape() as g:
                    conv_input, conv_output, Pred = layer_model(
                        class_data[c]['x'])
                    prob = Pred[:, c]
                    grads = g.gradient(prob, conv_output)
                    pooled_grads = K.mean(grads, axis=(0, 1, 2))
                    pooled_grads = tf.reshape(pooled_grads,
                                              shape=(_weights.shape[-2],
                                                     _weights.shape[-1]))
                s_weights = tf.reduce_mean(tf.abs(
                    tf.multiply(pooled_grads, _weights)),
                                           axis=(1, 2, 3))
                ax = fig.add_subplot(2, 2, c + 1)
                ax.set_xlabel('({}) {}'.format(chr(c + 97),
                                               self.class_names[c]))
                viz.plot_topomap(np.array(s_weights),
                                 self.locs,
                                 names=self.sensors_name,
                                 show_names=True,
                                 show=False,
                                 image_interp='bicubic',
                                 cmap='Spectral_r',
                                 extrapolate='head',
                                 sphere=(0, 0, 0, 1))  # draw topographic image
            plt.tight_layout(pad=0)
            plt.show(block=False)

    def fs_class_fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.input
        _output = self.model.output
        plt.rcParams['font.size'] = 12
        for name in layer_name:
            layer = self.layer_dict[name]
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            fig, axs = plt.subplots(2, 2, sharex=True)
            for c in class_data:
                with tf.GradientTape() as g:
                    conv_output, Pred = layer_model(class_data[c]['x'])
                    prob = Pred[:, c]
                    grads = g.gradient(prob, conv_output)
                    pooled_grads = K.mean(grads, axis=(0, 1, 2))
                selected = tf.multiply(pooled_grads, conv_output)
                fred = np.average(np.abs(fft(np.array(selected), axis=2)),
                                  axis=(0, 1, -1))
                fred = fred / len(fred.T)
                fred = fred[:101]
                axs[c // 2, c % 2].plot(np.arange(len(fred.T)), fred.T)
                axs[c // 2,
                    c % 2].set_xlabel('({}) {}'.format(chr(c + 97),
                                                       self.class_names[c]))
                axs[c // 2, c % 2].autoscale(enable=True,
                                             axis='both',
                                             tight=True)
            plt.subplots_adjust(right=0.97,
                                left=0.04,
                                top=0.96,
                                bottom=0.10,
                                wspace=0.13,
                                hspace=0.10)
            plt.show(block=False)

    def fs_class_freq_topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.input
        _output = self.model.output
        ib = ['2-8', '8-12', '12-20', '20-30', '30-60']
        plt.rcParams['font.size'] = 12
        for name in layer_name:
            layer = self.layer_dict[name]
            _weights = layer.get_weights()[0]
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            fig = plt.figure(figsize=(8,6))
            plt.Axes.add_child_axes
            gs = fig.add_gridspec(2, 2)
            for c in class_data:
                axs = fig.add_subplot(gs[c])
                ax = inset_axes(axs,
                                '100%',
                                '100%',
                                bbox_to_anchor=(0, 0.25, 1, 0.5),
                                bbox_transform=axs.transAxes,
                                borderpad=0)
                cax = inset_axes(axs,
                                 '100%',
                                 '100%',
                                 bbox_to_anchor=(0.65, 0.65, 0.09, 0.03),
                                 bbox_transform=axs.transAxes,
                                 borderpad=0)
                text_0 = inset_axes(axs,
                                    '100%',
                                    '100%',
                                    bbox_to_anchor=(0.62, 0.65, 0.03, 0.03),
                                    bbox_transform=axs.transAxes,
                                    borderpad=0)
                text_1 = inset_axes(axs,
                                    '100%',
                                    '100%',
                                    bbox_to_anchor=(0.74, 0.65, 0.03, 0.03),
                                    bbox_transform=axs.transAxes,
                                    borderpad=0)
                cax_text = inset_axes(axs,
                                      '100%',
                                      '100%',
                                      bbox_to_anchor=(0.77, 0.65, 0.23, 0.03),
                                      bbox_transform=axs.transAxes,
                                      borderpad=0)
                title = inset_axes(axs,
                                   '100%',
                                   '100%',
                                   bbox_to_anchor=(0, 0.65, 0.62, 0.03),
                                   bbox_transform=axs.transAxes,
                                   borderpad=0)
                self._ban_axis(axs)
                self._ban_axis(ax)
                self._ban_axis(cax)
                self._ban_axis(cax_text)
                self._ban_axis(text_0)
                self._ban_axis(text_1)
                self._ban_axis(title)
                ax.set_xlabel('({}) {}'.format(chr(c + 97),
                                               self.class_names[c]))
                ibclass_data = interestingband(class_data[c]['x'],
                                               axis=-2,
                                               swapaxes=False)
                cax_text.text(0.5,
                              0.5,
                              'contribution',
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=cax_text.transAxes)
                text_0.text(0,
                            0.45,
                            '0',
                            horizontalalignment='left',
                            verticalalignment='center',
                            transform=text_0.transAxes)
                text_1.text(1,
                            0.45,
                            '1',
                            horizontalalignment='right',
                            verticalalignment='center',
                            transform=text_1.transAxes)
                title.text(0.02,
                           0.5,
                           'Inter-band topomap of class',
                           horizontalalignment='left',
                           verticalalignment='center',
                           transform=title.transAxes)
                for i in np.arange(ibclass_data.shape[0]):
                    with tf.GradientTape() as g:
                        conv_output, Pred = layer_model(ibclass_data[i])
                        prob = Pred[:, c]
                        grads = g.gradient(prob, conv_output)
                    pooled_grads = K.mean(grads, axis=(0, 1, 2))
                    pooled_grads = tf.reshape(pooled_grads,
                                              shape=(_weights.shape[-2],
                                                     _weights.shape[-1]))
                    s_weights = normalization(
                        np.mean(np.abs(np.array(pooled_grads * _weights)),
                                axis=(1, 2, 3)))
                    width = 1. / ibclass_data.shape[0]
                    ax_i = inset_axes(ax,
                                      '100%',
                                      '100%',
                                      bbox_to_anchor=(0 + width * i, 0, width,
                                                      1),
                                      bbox_transform=ax.transAxes,
                                      borderpad=0)
                    ax_i.set_xlabel('{}'.format(ib[i] + 'Hz'))
                    # self._ban_axis(ax_i)
                    im, cn = viz.plot_topomap(
                        s_weights,
                        self.locs,
                        vmax=1,
                        vmin=0,
                        names=self.sensors_name,
                        show_names=True,
                        show=False,
                        image_interp='bicubic',
                        cmap='RdBu_r',
                        extrapolate='head',
                        sphere=(0, 0, 0, 1))  # draw topographic image
                cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
                cbar.set_ticks([])
            plt.tight_layout(pad=0.25, h_pad=0, w_pad=1)
            plt.show(block=False)