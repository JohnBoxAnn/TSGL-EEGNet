# coding:utf-8
from scipy import stats
import numpy as np

x = np.array([76.00, 56.50, 81.25, 61.00, 55.00, 45.25, 82.75, 81.75, 70.75])
y = np.array([0.86, 0.24, 0.70, 0.68, 0.36, 0.34, 0.66, 0.75, 0.82])
a = np.array([85.41, 70.67, 95.24, 80.26, 70.29, 68.37, 90.97, 86.35, 84.47])
b = np.array([0.8054, 0.6091, 0.9365, 0.7361, 0.6044, 0.5779, 0.8797, 0.8180, 0.7929])

# x=np.array([80.56,65.44,65.97,99.32,89.19,86.11,81.25,88.82,86.81])
# y=np.array([76.39,61.03,60.41,98.65,85.14,80.56,78.47,84.21,81.94])
# z=np.array([66.56,57.81,61.25,94.06,80.63,75.00,72.50,89.38,85.63])

def p_value(x, y):
    t, t_p = stats.ttest_ind(x - y, np.zeros_like(y), equal_var=False)
    print('T test: t={}, p={}'.format(t, t_p))

    ks, ks_p = stats.ks_2samp(x - y, np.zeros_like(x))
    print('KS test: ks={}, p={}'.format(ks, ks_p))

    return (t, t_p), (ks, ks_p)

if __name__ == '__main__':
    p_value(b, y)