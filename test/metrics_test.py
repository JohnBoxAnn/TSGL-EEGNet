import __init__
import tensorflow as tf

from core.metrics import Kappa

print(tf.version.VERSION)
k = Kappa(4)
print(k.reset_states)