# coding:utf-8
from scipy import stats
import numpy as np

x = np.array([83.77, 70.18, 94.36, 75.88, 64.35, 65.67, 88.95, 83.84, 83.64])
y = np.array([84.52, 61.17, 95.90, 66.58, 60.22, 56.65, 85.78, 84.83, 78.90])
z = np.array([82.70, 61.84, 92.67, 58.77, 55.51, 53.58, 78.99, 82.80, 82.05])
a = np.array([82.70, 61.84, 92.67, 58.77, 55.51, 53.58, 78.99, 82.80, 82.05])

# x=np.array([80.56,65.44,65.97,99.32,89.19,86.11,81.25,88.82,86.81])
# y=np.array([76.39,61.03,60.41,98.65,85.14,80.56,78.47,84.21,81.94])
# z=np.array([66.56,57.81,61.25,94.06,80.63,75.00,72.50,89.38,85.63])

def p_value(x, y):
    t, t_p = stats.ttest_ind(x - y, np.zeros_like(x), equal_var=False)
    print('T test: t={}, p={}'.format(t, t_p))

    ks, ks_p = stats.ks_2samp(x - y, np.zeros_like(x))
    print('KS test: ks={}, p={}'.format(ks, ks_p))

    return (t, t_p), (ks, ks_p)

if __name__ == '__main__':
    p_value(x, z)