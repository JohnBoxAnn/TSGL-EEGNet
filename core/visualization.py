# coding:utf-8

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import mne.viz as viz
from numbers import Integral, Real
from tensorflow.python.keras.api._v2.keras import backend as K
from scipy.fftpack import fft
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from core.utils import interestingband, normalization
from core.utils import load_locs, azim_proj
from core.generators import BaseGenerator as _BaseGenerator


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
                 dataGen: _BaseGenerator,
                 srate=None,
                 vis_data=None,
                 locpath=None):
        self.model = model
        self.layer_dict = dict([(layer.name, layer)
                                for layer in model.layers[1:]])
        if not isinstance(dataGen, _BaseGenerator):
            raise TypeError('`dataGen` should be passed as an instance '
                            'of `BaseGenerator`.')
        if not isinstance(srate, Real):
            raise TypeError('`srate` should be passed as a real number.')
        self.srate = srate
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
        a = []
        b = []
        [(a.append(x), b.append(y))
         for (x, y) in dataGen(self.data['x'], self.data['y'])]
        self.data = {'x': np.concatenate(a), 'y': np.concatenate(b)}
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
    def _ban_axis(ax: plt.Axes):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

    def _check_layer_name(self, layer_name):
        if not isinstance(layer_name, (list, tuple)):
            if isinstance(layer_name, str):
                layer_name = [layer_name]
            else:
                raise TypeError('`layer_name` should be passed as a string'
                                ' or a list of string')
        for name in layer_name:
            if not isinstance(name, str):
                raise TypeError('`layer_name` should be passed as a string'
                                ' or a list of string')
            if not name in self.layer_dict.keys():
                raise ValueError(
                    'No layer in the model named {} given in `layer_name`'.
                    format(name))
        return layer_name

    @staticmethod
    def show():
        plt.show()  # block the thread

    def fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.layers[0].input
        plt.rcParams['font.size'] = 16
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            r = t(self.data['x'])  # ndarray
            a = []
            fig = plt.figure(figsize=(16, 5))
            gs = fig.add_gridspec(
                _output.shape[-1] // 2 + _output.shape[-1] % 2, 2)
            for i in np.arange(_output.shape[-1]):
                fred = np.abs(fft(r[:, :, :, i], n=self.srate))
                fred = np.average(fred, axis=0)
                fred = fred / len(fred.T)
                fred = fred[:, 2:min(self.srate // 2 + 1, 101)]
                a.append(fred)
            a = normalization(np.array(a), axis=None)
            a = [a[i, :] for i in range(a.shape[0])]
            for i in np.arange(_output.shape[-1]):
                ax = fig.add_subplot(gs[i // 2, i % 2])
                if i == 1:
                    ax_legend = ax
                fred = a[i]
                ax.set_prop_cycle('color', [
                    plt.cm.Spectral_r(i) for i in np.linspace(0, 1, len(fred))
                ])
                for col in np.arange(len(fred)):
                    ax.plot(np.arange(2,
                                      len(fred.T) + 2),
                            fred[col],
                            label='electrode {:d}'.format(col + 1))
                text = inset_axes(ax,
                                  '100%',
                                  '100%',
                                  bbox_to_anchor=(0.85, 0.65, 0.1, 0.3),
                                  bbox_transform=ax.transAxes,
                                  borderpad=0)
                self._ban_axis(text)
                text.text(0.5,
                          0,
                          'Filter (' + chr(ord('a') + i) + ')',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=text.transAxes)
                ax.autoscale(enable=True, axis='both', tight=True)
                ax.set_xticks([2, 20, 40, 60, 80, 100])
                ax.set_xticklabels(['2', '20', '40', '60', '80', '100'],
                                   fontsize=12)
                if i // 2 == _output.shape[-1] // 2 - 1:
                    ax.set_xlabel('Frequency /Hz')
                else:
                    ax.set_xticklabels([])
                ax.set_yticks([0, 1.0])
                ax.set_yticklabels(['0', '1'], fontsize=12)
                if i % 2 == 0:
                    ax.set_ylabel('AMP')
                else:
                    ax.set_yticklabels([])
            ax_legend.legend(loc=2,
                             ncol=2,
                             bbox_to_anchor=(1.0325, 1.0),
                             borderaxespad=0.,
                             bbox_transform=ax_legend.transAxes)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.subplots_adjust(right=0.7325,
                                left=0.0325,
                                top=0.985,
                                bottom=0.11725,
                                wspace=0.05,
                                hspace=0.33)
            plt.margins(0, 0)
            fig.savefig(os.path.join('fft_output.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    def output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.layers[0].input
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            r = t(self.data['x'][0:1])  # ndarray
            fig = plt.figure()
            for i in range(_output.shape[-1]):
                rci = np.squeeze(r[:, :, :, i], axis=0)
                plt.subplot(_output.shape[-1], 1, i + 1)
                plt.plot(np.arange(len(rci.T)), rci.T)
                plt.autoscale(enable=True, axis='both', tight=True)
                plt.tight_layout()
            plt.tight_layout()
            plt.margins(0, 0)
            fig.savefig(os.path.join('output.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    def topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        plt.rcParams['font.size'] = 16
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            _min = np.min(_weights)
            _max = np.max(_weights)
            n = 1
            fig = plt.figure(figsize=(16, 9))
            gs = fig.add_gridspec(
                (_weights.shape[-2] * _weights.shape[-1] // 20) * 4 + 1, 20)
            for i in range(_weights.shape[-2]):
                for j in range(_weights.shape[-1]):
                    ax = fig.add_subplot(
                        gs[((n - 1) // 20) * 4:((n - 1) // 20 + 1) * 4,
                           (n - 1) % 20])
                    self._ban_axis(ax)
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
                        ax.spines['bottom'].set_position(('axes', 1.3))
                        if n <= 10:
                            ax.set_xlabel(n, fontsize=16)
                        else:
                            ax.set_xlabel(n - 10, fontsize=16)
                    if n % 20 == 1:
                        ax.spines['left'].set_position(('axes', 0.08))
                        ax.set_ylabel('(' + chr(ord('a') + n // 10) + ')',
                                      fontsize=16,
                                      rotation=90)
                    if n % 20 == 0:
                        ax.spines['left'].set_position(('axes', 1.35))
                        ax.set_ylabel('(' + chr(ord('a') - 1 + n // 10) + ')',
                                      fontsize=16,
                                      rotation=90)
                    n += 1
            axs = fig.add_subplot(
                gs[(_weights.shape[-2] * _weights.shape[-1] // 20) * 4, :])
            self._ban_axis(axs)
            divider = make_axes_locatable(axs)
            cax = divider.append_axes('top', size='10%', pad=1)
            cb = fig.colorbar(im,
                              cax=cax,
                              orientation='horizontal',
                              ticklocation='bottom')
            plt.subplots_adjust(right=0.98,
                                left=0.02,
                                top=0.99,
                                bottom=0,
                                wspace=0,
                                hspace=0)
            plt.margins(0, 0)
            fig.savefig(os.path.join('topo_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    def fft_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            fig, axs = plt.subplots()
            rci = np.squeeze(_weights)
            fred = np.abs(fft(rci.T))
            fred = fred / len(fred.T)
            fred = fred[:, 2:int((len(fred.T) / 2)) + 1]
            axs.set_title('Weights in layer {} after FFT'.format(name))
            axs.set_prop_cycle('color', [
                plt.cm.Spectral_r(i) for i in np.linspace(0, 1, fred.shape[0])
            ])
            axs.plot(np.arange(2, len(fred.T) + 2), fred.T)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.tight_layout()
            fig.savefig(os.path.join('fft_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    def line_kernel(self, layer_name, lines: list = None):
        layer_name = self._check_layer_name(layer_name)
        plt.rcParams['font.size'] = 16
        for name in layer_name:
            _weights = self.layer_dict[name].get_weights()[0]
            print(_weights.shape)
            if lines is None:
                lines = np.arange(_weights.shape[-1])
            fig, axs = plt.subplots()
            axs.set_title('Weights in layer {}'.format(name))
            axs.set_prop_cycle(
                'color',
                [plt.cm.Spectral_r(i) for i in np.linspace(0, 1, len(lines))])
            axs.plot(np.squeeze(_weights)[:, lines])
            plt.subplots_adjust(right=0.99,
                                left=0.06,
                                top=0.93,
                                bottom=0.07,
                                wspace=0,
                                hspace=0)
            plt.margins(0, 0)
            fig.savefig(os.path.join('line_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
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
            plt.margins(0, 0)
            fig.savefig(os.path.join('kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
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
            plt.margins(0, 0)
            fig.savefig(os.path.join('depthwise_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    # NOTE: fs visualiztion
    def fs_topo_kernel(self, layer_name):
        layer_name = self._check_layer_name(layer_name)

    def fs_fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        _input = self.model.input
        _output = self.model.output
        for name in layer_name:
            layer = self.layer_dict[name]
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            with tf.GradientTape() as g:
                conv_output, Pred = layer_model(self.data['x'])
                index = np.argmax(Pred[0])
                prob = Pred[:, index]
                grads = g.gradient(prob, conv_output)
                pooled_grads = K.sum(grads, axis=(0, 1, 2))
            selected = tf.reduce_mean(tf.multiply(pooled_grads, conv_output),
                                      axis=0)
            a = []
            for i in np.arange(selected.shape[-1]):
                fred = np.abs(
                    fft(np.array(selected[:, :, i]), n=self.srate, axis=1))
                fred = fred / len(fred.T)
                fred = fred[:, 2:min(self.srate // 2 + 1, 101)]
                a.append(fred)
            a = normalization(np.array(a), axis=None)
            a = [a[i, :] for i in range(a.shape[0])]
            fig = plt.figure(figsize=(16, 5))
            gs = fig.add_gridspec(
                selected.shape[-1] // 2 + selected.shape[-1] % 2, 2)
            for i in np.arange(selected.shape[-1]):
                ax = fig.add_subplot(gs[i // 2, i % 2])
                ax.set_prop_cycle('color', [
                    plt.cm.Spectral_r(i) for i in np.linspace(0, 1, len(fred))
                ])
                if i == 1:
                    ax_legend = ax
                fred = a[i]
                for col in np.arange(len(fred)):
                    ax.plot(np.arange(2,
                                      len(fred.T) + 2),
                            fred[col],
                            label='electrode {:d}'.format(col + 1))
                text = inset_axes(ax,
                                  '100%',
                                  '100%',
                                  bbox_to_anchor=(0.85, 0.65, 0.1, 0.3),
                                  bbox_transform=ax.transAxes,
                                  borderpad=0)
                self._ban_axis(text)
                text.text(0.5,
                          0,
                          'Filter (' + chr(ord('a') + i) + ')',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=text.transAxes)
                ax.autoscale(enable=True, axis='both', tight=True)
                ax.set_xticks([2, 20, 40, 60, 80, 100])
                ax.set_xticklabels(['2', '20', '40', '60', '80', '100'],
                                   fontsize=12)
                if i // 2 == selected.shape[-1] // 2 - 1:
                    ax.set_xlabel('Frequency /Hz')
                else:
                    ax.set_xticklabels([])
                ax.set_yticks([0, 1.0])
                ax.set_yticklabels(['0', '1'], fontsize=12)
                if i % 2 == 0:
                    ax.set_ylabel('AMP')
                else:
                    ax.set_yticklabels([])
            ax_legend.legend(loc=2,
                             ncol=2,
                             bbox_to_anchor=(1.0325, 1.0),
                             borderaxespad=0.,
                             bbox_transform=ax_legend.transAxes)
            plt.autoscale(enable=True, axis='both', tight=True)
            plt.subplots_adjust(right=0.7325,
                                left=0.0325,
                                top=0.985,
                                bottom=0.11725,
                                wspace=0.05,
                                hspace=0.33)
            plt.margins(0, 0)
            fig.savefig(os.path.join('fs_fft_output.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    # NOTE: class visualization
    def class_fft_output(self, layer_name):
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.layers[0].input
        plt.rcParams['font.size'] = 12
        for name in layer_name:
            _output = self.layer_dict[name].output
            t = K.function(_input, _output)
            fig, axs = plt.subplots()
            unselected = []
            for c in class_data:
                unselected.append(t(class_data[c]['x']))
            axs.set_prop_cycle('color', 'rbgy')
            a = []
            for c in class_data:
                fred = np.mean(np.abs(
                    fft(np.array(unselected[c]), n=self.srate, axis=2)),
                               axis=(0, 1, -1))
                fred = fred / len(fred.T)
                fred = fred[2:min(self.srate // 2 + 1, 101)]
                a.append(fred)
            a = normalization(np.array(a), axis=None)
            a = [a[i, :] for i in range(a.shape[0])]
            for c in class_data:
                fred = a[c]
                axs.plot(np.arange(2,
                                   len(fred.T) + 2),
                         fred.T,
                         label='({}) {}'.format(chr(c + 97),
                                                self.class_names[c]))
                axs.set_xlabel('Frequency /Hz')
                axs.set_ylabel('Amplitude')
                axs.set_xticks([2, 20, 40, 60, 80, 100])
                axs.set_xticklabels(['2', '20', '40', '60', '80', '100'],
                                    fontsize=12)
                axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                                    fontsize=12)
            axs.autoscale(enable=True, axis='both', tight=True)
            plt.legend(loc='upper right')
            plt.tight_layout(pad=0.25, h_pad=0, w_pad=1)
            fig.savefig(os.path.join('class_fft_output.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
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
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            fig = plt.figure()
            for c in class_data:
                with tf.GradientTape() as g:
                    conv_output, Pred = layer_model(class_data[c]['x'])
                    prob = Pred[:, c]
                    grads = g.gradient(prob, conv_output)
                    pooled_grads = K.sum(grads, axis=(0, 1, 2))
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
            plt.margins(0, 0)
            fig.savefig(os.path.join('fs_class_topo_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
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
            fig, axs = plt.subplots()
            selected = []
            for c in class_data:
                with tf.GradientTape() as g:
                    conv_output, Pred = layer_model(class_data[c]['x'])
                    prob = Pred[:, c]
                    grads = g.gradient(prob, conv_output)
                pooled_grads = K.sum(grads, axis=(0, 1, 2))
                selected.append(tf.multiply(pooled_grads, conv_output))
            a = []
            for c in class_data:
                fred = np.mean(np.abs(
                    fft(np.array(selected[c]), n=self.srate, axis=2)),
                               axis=(0, 1, -1))
                fred = fred / len(fred.T)
                fred = fred[2:min(self.srate // 2 + 1, 101)]
                a.append(fred)
            a = normalization(np.array(a), axis=None)
            a = [a[i, :] for i in range(a.shape[0])]
            axs.set_prop_cycle('color', 'rbgy')
            for c in class_data:
                fred = a[c]
                axs.plot(np.arange(2,
                                   len(fred.T) + 2),
                         fred.T,
                         label='({}) {}'.format(chr(c + 97),
                                                self.class_names[c]))
                axs.set_xlabel('Frequency /Hz')
                axs.set_ylabel('Amplitude')
                axs.set_xlabel('Frequency /Hz')
                axs.set_ylabel('Amplitude')
                axs.set_xticks([2, 20, 40, 60, 80, 100])
                axs.set_xticklabels(['2', '20', '40', '60', '80', '100'],
                                    fontsize=12)
                axs.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
                axs.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'],
                                    fontsize=12)
            axs.autoscale(enable=True, axis='both', tight=True)
            plt.legend(loc='upper right')
            plt.tight_layout(pad=0.25, h_pad=0, w_pad=1)
            fig.savefig(os.path.join('fs_class_fft_output.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
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
            fig = plt.figure(figsize=(8, 6))
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
                                               ib,
                                               axis=-2,
                                               swapaxes=False)
                cax_text.text(0.5,
                              0.5,
                              'contribution',
                              horizontalalignment='center',
                              verticalalignment='center',
                              transform=cax_text.transAxes)
                text_0.text(0,
                            0.5,
                            '0',
                            horizontalalignment='left',
                            verticalalignment='center',
                            transform=text_0.transAxes)
                text_1.text(1,
                            0.5,
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
                s_weights = []
                for i in np.arange(ibclass_data.shape[0]):
                    with tf.GradientTape() as g:
                        conv_output, Pred = layer_model(ibclass_data[i])
                        prob = Pred[:, c]
                        grads = g.gradient(prob, conv_output)
                    pooled_grads = K.sum(grads, axis=(0, 1, 2))
                    pooled_grads = tf.reshape(pooled_grads,
                                              shape=(_weights.shape[-2],
                                                     _weights.shape[-1]))
                    s_weights.append(
                        np.sum(np.maximum(np.array(pooled_grads * _weights),
                                           0),
                                axis=(1, 2, 3)))
                s_weights = normalization(np.array(s_weights), axis=None)
                s_weights = [
                    s_weights[i, :] for i in range(s_weights.shape[0])
                ]
                for i in np.arange(len(s_weights)):
                    width = 1. / len(s_weights)
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
                        s_weights[i],
                        self.locs,
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
            fig.savefig(os.path.join('fs_class_freq_topo_kernel.png'),
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)

    def fs_onetrial_freq_topo_kernel(self, layer_name, data):
        '''this function can save figures to disk, return classification and figure path'''
        layer_name = self._check_layer_name(layer_name)
        class_data = self._class_data(self.data)
        _input = self.model.input
        _output = self.model.output
        ib = ['8-12', '12-24']  # interesting bands (alpha & beta)
        plt.rcParams['font.size'] = 16  # set font
        for name in layer_name:
            layer = self.layer_dict[name]
            _weights = layer.get_weights()[0]
            layer_model = tf.keras.Model([_input], [layer.output, _output])
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(1, 1)
            axs = fig.add_subplot(gs[0])
            ax = inset_axes(axs,
                            '100%',
                            '100%',
                            bbox_to_anchor=(0, 0.25, 1, 0.5),
                            bbox_transform=axs.transAxes,
                            borderpad=0)
            cax = inset_axes(axs,
                             '100%',
                             '100%',
                             bbox_to_anchor=(0.7, 0.75, 0.1, 0.03),
                             bbox_transform=axs.transAxes,
                             borderpad=0)
            text_0 = inset_axes(axs,
                                '100%',
                                '100%',
                                bbox_to_anchor=(0.68, 0.75, 0.02, 0.03),
                                bbox_transform=axs.transAxes,
                                borderpad=0)
            text_1 = inset_axes(axs,
                                '100%',
                                '100%',
                                bbox_to_anchor=(0.8, 0.75, 0.02, 0.03),
                                bbox_transform=axs.transAxes,
                                borderpad=0)
            cax_text = inset_axes(axs,
                                  '100%',
                                  '100%',
                                  bbox_to_anchor=(0.82, 0.75, 0.18, 0.03),
                                  bbox_transform=axs.transAxes,
                                  borderpad=0)
            title = inset_axes(axs,
                               '100%',
                               '100%',
                               bbox_to_anchor=(0, 0.75, 0.68, 0.03),
                               bbox_transform=axs.transAxes,
                               borderpad=0)
            self._ban_axis(axs)
            self._ban_axis(ax)
            self._ban_axis(cax)
            self._ban_axis(cax_text)
            self._ban_axis(text_0)
            self._ban_axis(text_1)
            self._ban_axis(title)
            # ax.set_xlabel('({}) {}'.format(chr(c + 97),
            #                                 self.class_names[c]))
            ibclass_data = interestingband(data[0],
                                           ib,
                                           axis=-2,
                                           swapaxes=False)
            cax_text.text(0.5,
                          0.5,
                          'contribution',
                          horizontalalignment='center',
                          verticalalignment='center',
                          transform=cax_text.transAxes)
            text_0.text(0,
                        0.5,
                        '0',
                        horizontalalignment='left',
                        verticalalignment='center',
                        transform=text_0.transAxes)
            text_1.text(1,
                        0.5,
                        '1',
                        horizontalalignment='right',
                        verticalalignment='center',
                        transform=text_1.transAxes)
            title.text(0.02,
                       0.5,
                       'Topomaps of Interested-bands (alpha, beta)',
                       horizontalalignment='left',
                       verticalalignment='center',
                       transform=title.transAxes)
            s_weights = []
            for i in np.arange(ibclass_data.shape[0]):
                with tf.GradientTape() as g:
                    conv_output, Pred = layer_model(ibclass_data[i])
                    prob = Pred[:, np.argmax(Pred)]
                    grads = g.gradient(prob, conv_output)
                pooled_grads = K.sum(grads, axis=(0, 1, 2))
                pooled_grads = tf.reshape(pooled_grads,
                                          shape=(_weights.shape[-2],
                                                 _weights.shape[-1]))
                s_weights.append(
                    np.mean(np.array(K.maximum(pooled_grads * _weights)),
                            axis=(1, 2, 3)))
            s_weights = normalization(np.array(s_weights), axis=None)
            s_weights = [s_weights[i, :] for i in range(s_weights.shape[0])]
            for i in np.arange(len(s_weights)):
                width = 1. / len(s_weights)
                ax_i = inset_axes(ax,
                                  '100%',
                                  '100%',
                                  bbox_to_anchor=(0 + width * i, 0, width, 1),
                                  bbox_transform=ax.transAxes,
                                  borderpad=0)
                ax_i.set_xlabel('{}'.format(ib[i] + 'Hz'))
                # self._ban_axis(ax_i)
                im, cn = viz.plot_topomap(s_weights[i],
                                          self.locs,
                                          names=self.sensors_name,
                                          show_names=True,
                                          show=False,
                                          image_interp='bicubic',
                                          cmap='RdBu_r',
                                          extrapolate='head',
                                          sphere=(0, 0, 0,
                                                  1))  # draw topographic image
            cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
            cbar.set_ticks([])
            plt.tight_layout(pad=0.25, h_pad=0, w_pad=1)
            # saving figure to disk
            figurepath = os.path.join('fs_onetrial_freq_topo_kernel.png')
            fig.savefig(figurepath,
                        format='png',
                        transparent=False,
                        dpi=300,
                        pad_inches=0)
            plt.show(block=False)
            # return classification and figure path
            return np.argmax(Pred), figurepath