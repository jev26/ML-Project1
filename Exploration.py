from proj1_helpers import load_csv_data
import matplotlib.pyplot as plt
import numpy as np
from implementation import *

import random



#TODO: documentation
def data_cleaning(tX, i):
    #exclude values = -999
    tX_index_wash1 = tX[:, i] != -999
    tX_wash1 = tX[tX_index_wash1, i]

    #exclude value > 5*standard deviation
    tX_wash1_std = np.std(tX_wash1)
    tX_index_wash2 = (tX_wash1 <= tX_wash1_std * 5) & (-tX_wash1_std * 5 <= tX_wash1)
    tX_tmp = tX_wash1[tX_index_wash2]


    y_tmp1 = y[tX_index_wash1]
    y_tmp = y_tmp1[tX_index_wash2]

    return tX_tmp, y_tmp




data_path = 'data/train.csv'
y, tX, ids = load_csv_data(data_path, sub_sample=False)

# print(y.shape) => (250000,)
# print(tX.shape) => (250000, 30) => 30 features
# print(ids.shape) => (250000,)

nSample, nFeature = tX.shape

"""
data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)

print(data1[:,1].shape)
"""
# count the number of -999 per feature
"""
for i in range(nFeature):
    tmp = sum(tX[:, i] == -999)
    print('Feature n° ' + str(i) + " " + str(tmp))
    print('Ratio ' + str(tmp/nSample))

# count the number of 0 per feature
for i in range(nFeature):
    tmp = sum(tX[:, i] == 0)
    print('Feature n° ' + str(i) + " " + str(tmp))
    print('Ratio ' + str(tmp/nSample))
"""

# feature ranking according to the correlation between feature and label
damagedFeature = np.array([5, 12, 26, 27, 28, 23, 24, 25, 22, 29])
cor = np.zeros(nFeature)

## correlation when -999 are dropped
cor = np.zeros(nFeature)
print(nSample)
for iFeature in range(nFeature):

    tX_tmp, y_tmp = data_cleaning(tX, iFeature)

    tmp = np.corrcoef(tX_tmp,y_tmp)**2
    #print(tmp)
    cor[iFeature] = tmp[1][0]

orderedPower = -np.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])

#print(orderedPower)
#print(orderedInd)


# for each feature, display the histogram with color = label (y)

damagedFeature0 = orderedInd
for i in range(len(damagedFeature0)):

    tX_tmp, y_tmp = data_cleaning(tX, damagedFeature0[i])

    maximum = max(tX_tmp)
    minimum = min(tX_tmp)

    data1 = tX_tmp[y_tmp == -1]
    data2 = tX_tmp[(y_tmp == 1)]


    #plot the figure

    bins = np.linspace(minimum, maximum, 80)
    #plt.figure(i)
    #plt.hist(data1, bins, alpha=0.5, label='x')
    #plt.hist(data2, bins, alpha=0.5, label='y')
    #plt.xlim(minimum,maximum)
    #plt.legend(loc='upper right')
    #plt.title('Feature n° ' + str(damagedFeature0[i]))
    #plt.show()
    lambda_ = np.linspace(-2,2,40)
    degree = np.linspace(1,12,12)
    mse_losses = np.zeros(degree.shape)
    for j in range(degree.size):
        poly_tX = build_poly(tX_tmp, degree[j].astype(int))
        weights, mse_loss = ridge_regression(y_tmp, poly_tX, 0.5)
        mse_losses[j] = mse_loss
    plt.figure()
    plt.plot(degree,mse_losses)
    plt.title('mseLoss in function of Lambda')
    plt.show()

