import os
import logging
import numpy as np
import matplotlib.pyplot as plt


def statistic_best(val_loss, val_acc, p=0.05, upstep=301):
    acc_op = np.greater
    loss_op = np.less
    best_acc = -np.Inf
    best_loss = np.Inf
    for i in np.arange(0, min(len(val_loss), upstep)):
        current_loss = val_loss[i]
        current_acc = val_acc[i]
        if current_loss is None or current_acc is None:
            logging.warning('val_loss or val_accuracy is unavailable, skipping.')
        else:
            if loss_op(np.abs(best_loss - current_loss),
                       current_loss * p / 2.):
                if acc_op(current_acc, best_acc):
                    best_loss = current_loss
                    best_acc = current_acc
                else:
                    val_loss[i] = best_loss
                    val_acc[i] = best_acc
            elif loss_op(current_loss, best_loss):
                best_loss = current_loss
                best_acc = current_acc
            else:
                val_loss[i] = best_loss
                val_acc[i] = best_acc
    return val_loss[:upstep], val_acc[:upstep]


filepath1 = os.path.join(
    'model\TestTSGLEEGNetA\\03\l1(0.00010000)_l21(0.00007500)_tl1(0.00000250)_F(16)_D(10)_Ns(20)_A03T_rawEEGConvNet(4).npy'
)
filepath2 = os.path.join(
    'model\CV_2021_03_04_19_01_28_EEGAttentionNet\F(09)_D(04)_A03T_EEGAttentionNet(1).npy'
)
history1 = np.load(file=filepath1, allow_pickle=True)
history1 = eval(str(history1))
history2 = np.load(file=filepath2, allow_pickle=True)
history2 = eval(str(history2))

val_loss1, val_acc1 = statistic_best(history1['val_loss'],
                                     history1['val_accuracy'])
val_loss2, val_acc2 = statistic_best(history2['val_loss'],
                                     history2['val_accuracy'])
fig = plt.figure()
# plt.plot(np.arange(0, len(val_loss1)), val_loss1, '--g')
plt.plot(np.arange(0, len(val_acc1)), val_acc1, '-g')
# plt.plot(np.arange(0, len(val_loss2)), val_loss2, '--r')
plt.plot(np.arange(0, len(val_acc2)), val_acc2, '-r')
plt.legend(['TSGL-EEGNet', 'EEGAttentionNet'])
plt.xlabel('Steps')
plt.ylabel('Accuracy')
plt.autoscale(enable=True, axis='x', tight=True)
plt.tight_layout()
fig.savefig(os.path.join('history.png'),
            format='png',
            transparent=False,
            dpi=300,
            pad_inches=0)
plt.show()
