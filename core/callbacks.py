# coding:utf-8
import logging
import numpy as np

from tensorflow.python.keras.api._v2.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.api._v2.keras.callbacks import EarlyStopping
from tensorflow.python.keras.api._v2.keras.callbacks import Callback


class StatModelCheckpoint(ModelCheckpoint):
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 **kwargs):
        super().__init__(filepath,
                         monitor=monitor,
                         verbose=verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         mode=mode,
                         save_freq=save_freq,
                         **kwargs)
        if 'p' in kwargs:
            self.p = float(kwargs['p'])
            if self.p <= 0. or self.p >= 1.:
                raise ValueError('`p` must above 0 and below 1.')
        else:
            self.p = 0.

        if 'statistic_best' in kwargs:
            self.statistic_best = kwargs['statistic_best']
            if isinstance(self.statistic_best, bool):
                if not self.statistic_best and self.p:
                    logging.warning('`p` argument is active only when '
                                    '`statistic_best` = True.')
                if not save_best_only:
                    logging.warning('`statistic_best` argument is active'
                                    ' only when `save_best_only` = True.')
            else:
                raise TypeError('`statistic_best` must be bool.')
        else:
            self.statistic_best = False

        if self.statistic_best:
            self.acc_op = np.greater
            self.loss_op = np.less
            self.best_acc = -np.Inf
            self.best_loss = np.Inf

    def _save_model(self, epoch, logs):
        """Saves the model.

        Arguments:
            epoch: the epoch this iteration is in.
            logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
        """
        logs = logs or {}

        if isinstance(self.save_freq,
                      int) or self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self._get_file_path(epoch, logs)

            try:
                if self.save_best_only:
                    if self.statistic_best:
                        current_loss = logs.get('val_loss')
                        current_acc = logs.get('val_accuracy')
                        if current_loss is None or current_acc is None:
                            logging.warning(
                                'Can save best model only with val_loss'
                                ' and val_accuracy available, skipping.')
                        else:
                            if self.loss_op(
                                    np.abs(self.best_loss - current_loss),
                                    current_loss * self.p / 2.):
                                if self.acc_op(current_acc, self.best_acc):
                                    if self.verbose > 0:
                                        print(
                                            '\nEpoch %05d: %s changed from %0.5f to %0.5f '
                                            'unsignificantly with p = %0.2f, but %s improved'
                                            ' from %0.5f to %0.5f, saving model'
                                            % (epoch + 1, 'val_loss',
                                               self.best_loss, current_loss,
                                               self.p, 'val_accuracy',
                                               self.best_acc, current_acc))
                                    self.best_loss = current_loss
                                    self.best_acc = current_acc
                                    if self.save_weights_only:
                                        self.model.save_weights(filepath,
                                                                overwrite=True)
                                    else:
                                        self.model.save(filepath,
                                                        overwrite=True)
                                else:
                                    if self.verbose > 0:
                                        print(
                                            '\nEpoch %05d: %s did not improve from %0.5f '
                                            'significantly with p = %0.2f and %s did not'
                                            ' improve from %0.5f' %
                                            (epoch + 1, 'val_loss',
                                             self.best_loss, self.p,
                                             'val_accuracy', self.best_acc))
                            elif self.loss_op(current_loss, self.best_loss):
                                if self.acc_op(self.best_acc - current_acc,
                                               20 * self.p * self.best_acc):
                                    if self.verbose > 0:
                                        print(
                                            '\nEpoch %05d: %s improved from %0.5f to %0.5f '
                                            'significantly with p = %0.2f, but %s reduced '
                                            'too much. Maybe it happens exploding gradient,'
                                            ' don\'t save.' %
                                            (epoch + 1, 'val_loss',
                                             self.best_loss, current_loss,
                                             self.p, 'val_acc'))
                                else:
                                    if self.verbose > 0:
                                        print(
                                            '\nEpoch %05d: %s improved from %0.5f to %0.5f '
                                            'significantly with p = %0.2f, saving model'
                                            % (epoch + 1, 'val_loss',
                                               self.best_loss, current_loss,
                                               self.p))
                                    self.best_loss = current_loss
                                    self.best_acc = current_acc
                                    if self.save_weights_only:
                                        self.model.save_weights(filepath,
                                                                overwrite=True)
                                    else:
                                        self.model.save(filepath,
                                                        overwrite=True)
                            else:
                                if self.verbose > 0:
                                    print(
                                        '\nEpoch %05d: %s did not improve from %0.5f'
                                        % (epoch + 1, 'val_loss',
                                           self.best_loss))
                    else:
                        current = logs.get(self.monitor)
                        if current is None:
                            logging.warning(
                                'Can save best model only with %s available, '
                                'skipping.', self.monitor)
                        else:
                            if self.monitor_op(current, self.best):
                                if self.verbose > 0:
                                    print(
                                        '\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                        ' saving model' %
                                        (epoch + 1, self.monitor, self.best,
                                         current))
                                self.best = current
                                if self.save_weights_only:
                                    self.model.save_weights(filepath,
                                                            overwrite=True)
                                else:
                                    self.model.save(filepath, overwrite=True)
                            else:
                                if self.verbose > 0:
                                    print(
                                        '\nEpoch %05d: %s did not improve from %0.5f'
                                        % (epoch + 1, self.monitor, self.best))
                else:
                    if self.verbose > 0:
                        print('\nEpoch %05d: saving model to %s' %
                              (epoch + 1, filepath))
                    if self.save_weights_only:
                        self.model.save_weights(filepath, overwrite=True)
                    else:
                        self.model.save(
                            filepath,
                            overwrite=True,
                        )

                self._maybe_remove_file()
            except IOError as e:
                # `e.errno` appears to be `None` so checking the content of `e.message`.
                if 'is a directory' in e.message:
                    raise IOError(
                        'Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))


class StatEarlyStopping(EarlyStopping):
    def __init__(self,
                 monitor='val_loss',
                 min_delta=0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False,
                 **kwargs):
        super().__init__(monitor=monitor,
                         min_delta=min_delta,
                         patience=patience,
                         verbose=verbose,
                         mode=mode,
                         baseline=baseline,
                         restore_best_weights=restore_best_weights)
        if 'p' in kwargs:
            self.p = float(kwargs['p'])
            if self.p <= 0. or self.p >= 1.:
                raise ValueError('`p` must above 0 and below 1.')
        else:
            self.p = 0.

        if 'statistic_best' in kwargs:
            self.statistic_best = kwargs['statistic_best']
            if isinstance(self.statistic_best, bool):
                if not self.statistic_best and self.p:
                    logging.warning('`p` argument is active only when '
                                    '`statistic_best` = True.')
            else:
                raise TypeError('`statistic_best` must be bool.')
        else:
            self.statistic_best = False

        if self.baseline and self.statistic_best:
            if not isinstance(self.baseline, (list, tuple)):
                raise ValueError('When using statistical earlystopping,'
                                 ' `baseline` should pass both loss and'
                                 ' acc sequentially.')

        if self.statistic_best:
            self.acc_op = np.greater
            self.loss_op = np.less
            self.best_acc = -np.Inf
            self.best_loss = np.Inf
            self.min_delta_loss = -abs(self.min_delta)
            self.min_delta_acc = abs(self.min_delta)

    def on_train_begin(self, logs=None):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        if self.statistic_best:
            if self.baseline is not None:
                self.best_loss = self.baseline[0]
                self.best_acc = self.baseline[1]
            else:
                self.best_acc = -np.Inf
                self.best_loss = np.Inf
        else:
            if self.baseline is not None:
                self.best = self.baseline
            else:
                self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        if self.statistic_best:
            current_loss = logs.get('val_loss')
            current_acc = logs.get('val_accuracy')
            if current_loss is None or current_acc is None:
                logging.warning(
                    'Early stopping conditioned on metric `%s` and `%s` '
                    'which are not available. Available metrics are: %s',
                    'val_loss', 'val_acc', ','.join(list(logs.keys())))
            else:
                if self.loss_op(
                        np.abs(self.best_loss - self.min_delta_loss -
                               current_loss), current_loss * self.p / 2.):
                    if self.acc_op(current_acc - self.min_delta_acc,
                                   self.best_acc):
                        self.best_loss = current_loss
                        self.best_acc = current_acc
                        self.wait = 0
                        if self.restore_best_weights:
                            self.best_weights = self.model.get_weights()
                    else:
                        self.wait += 1
                        if self.wait >= self.patience:
                            self.stopped_epoch = epoch
                            self.model.stop_training = True
                            if self.restore_best_weights:
                                if self.verbose > 0:
                                    print(
                                        'Restoring model weights from the end of the best epoch.'
                                    )
                                self.model.set_weights(self.best_weights)
                elif self.loss_op(current_loss - self.min_delta_loss,
                                  self.best_loss):
                    if self.acc_op(
                            self.best_acc - self.min_delta_acc - current_acc,
                            20 * self.p * self.best_acc):
                        self.wait += 1
                        if self.wait >= self.patience:
                            self.stopped_epoch = epoch
                            self.model.stop_training = True
                            if self.restore_best_weights:
                                if self.verbose > 0:
                                    print(
                                        'Restoring model weights from the end of the best epoch.'
                                    )
                                self.model.set_weights(self.best_weights)
                    else:
                        self.best_loss = current_loss
                        self.best_acc = current_acc
                        self.wait = 0
                        if self.restore_best_weights:
                            self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                        if self.restore_best_weights:
                            if self.verbose > 0:
                                print(
                                    'Restoring model weights from the end of the best epoch.'
                                )
                            self.model.set_weights(self.best_weights)
        else:
            super().on_epoch_end(epoch, logs)
