#coding:utf-8
import __init__
import os
import random

from core.generators import RawGenerator
from core.splits import StratifiedKFold

datadir = os.path.join('data', 'A')
kFold = 5
for root, dirs, files in os.walk(datadir):
    if files:
        dn = files[0][0]
        break
rg = RawGenerator(
    batch_size=10,
    kFold=kFold,
    #splitMethod=StratifiedKFold,
    datadir=datadir,
    dn=dn,
    random_state=random.randint(0, 1e8 - 1))
print(rg.random_state)

for k in range(1, kFold + 2):
    i = 0
    for x, y in rg(3, mode=RawGenerator.MODE_TRAIN):
        i += 1
        print(k, i, y[0], len(y))

# i = 0
# for x, y in rg(3, mode=rawGenerator.MODE_VAL):
#     i += 1
#     print(i, y[0], len(y))

# i = 0
# for x, y in rg(3, mode=rawGenerator.MODE_TEST):
#     i += 1
#     print(i, y[0], len(y))
