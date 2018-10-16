from proj1_helpers import load_csv_data
import matplotlib.pyplot as plt
import numpy as np
from implementation import *

import random

data_path = 'data/train.csv'
y, tX, ids = load_csv_data(data_path, sub_sample=False)

# print(y.shape) => (250000,)
# print(tX.shape) => (250000, 30) => 30 features
# print(ids.shape) => (250000,)

nSample, nFeature = tX.shape

data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)

print(data1[:,1].shape)

# count the number of -999 per feature
for i in range(nFeature):
    tmp = sum(tX[:, i] == -999)
    print('Feature n° ' + str(i) + " " + str(tmp))
    print('Ratio ' + str(tmp/nSample))

# count the number of 0 per feature
for i in range(nFeature):
    tmp = sum(tX[:, i] == 0)
    print('Feature n° ' + str(i) + " " + str(tmp))
    print('Ratio ' + str(tmp/nSample))

# feature ranking according to the correlation between feature and label
damagedFeature = np.array([5, 12, 26, 27, 28, 23, 24, 25, 22, 29])
cor = np.zeros(nFeature)
for iFeature in range(nFeature):
    # print(tX[:,iFeature])
    # print(y)
    if iFeature not in damagedFeature:
        tmp = np.corrcoef(tX[:,iFeature],y)**2
        cor[iFeature] = tmp[1][0]
orderedPower = -np.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])
print(orderedPower)
print(orderedInd)

# for each feature, display the histogram with color = label (y)

damagedFeature0 = orderedInd
for i in range(len(damagedFeature0)):

    maximum = max(tX[:,damagedFeature0[i]])
    minimum = min(tX[:,damagedFeature0[i]])

    print(i)
    x = data1[:,damagedFeature0[i]]
    y = data2[:,damagedFeature0[i]]

    bins = np.linspace(minimum, maximum, maximum-minimum)
    plt.figure(i)
    plt.hist(x, bins, alpha=0.5, label='x')
    plt.hist(y, bins, alpha=0.5, label='y')
    plt.legend(loc='upper right')
    plt.title('Feature n° ' + str(i))
    plt.show()

