from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *

import random

#TODO: documentation
def plot_feature(tX,orderedInd,y):
    for i in range(len(orderedInd)):
        non_altered_tX_tmp,y_tmp_initial_feature = data_cleaning(tX[:,orderedInd[i+1]],y)
        tX_tmp, y_tmp = data_cleaning((tX[:,orderedInd[i]]), y)
        #tX_tmp = (np.log(tX_tmp - min(tX_tmp)+1))
        tX_tmp = tX_tmp*non_altered_tX_tmp
        maximum = max(tX_tmp)
        minimum = min(tX_tmp)
        data1 = tX_tmp[y_tmp == -1]
        data2 = tX_tmp[y_tmp == 1]

    # plot the figure
        bins = np.linspace(minimum, maximum, 2000)

        plt.hist(data1, bins, alpha=0.5, label='x')
        plt.hist(data2, bins, alpha=0.5, label='y')
        plt.xlim(minimum, maximum)
        #plt.show()
        tmp = np.corrcoef(tX_tmp, y_tmp) ** 2
        non_altered_tmp = np.corrcoef(non_altered_tX_tmp,y_tmp)**2
        cor = tmp[1][0]
        non_altered_cor = non_altered_tmp[1][0]
        print(cor)
        print(non_altered_cor)

        if cor > non_altered_cor:
            print(orderedInd[i+1])
            print(orderedInd[i])
            print(True)

# count the number of 'value' per feature
def RatioCountValue(value,tX):
    # value is typically -999 or 0
    nSample, nFeature = tX.shape
    for i in range(nFeature):
        tmp = sum(tX[:, i] == value)
        print('Feature n° ' + str(i) + " " + str(tmp))
        print('Ratio ' + str(tmp/nSample))

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)

# print(y.shape) => (250000,)
# print(tX.shape) => (250000, 30) => 30 features
# print(ids.shape) => (250000,)

nSample, nFeature = tX.shape

"""
data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)
"""


"""
## Durty imputation
cleaned_data = tX
print(cleaned_data.shape)

case = 'imputation'

if case == 'remove':
    for iFeature in range(nFeature):
        # remove 999 values
        tmp_index = cleaned_data[:, iFeature] != -999
        cleaned_data = cleaned_data[tmp_index, :]
        y = y[tmp_index]

    #print(cleaned_data.shape)
    tX = cleaned_data

elif case == 'imputation':
    for iFeature in range(nFeature):
        # replace 999 values by the mean of the feature without considering the 999
        tmp_index = cleaned_data[:, iFeature] != -999
        tmp_vect = cleaned_data[tmp_index, iFeature]
        tmp_mean = np.mean(tmp_vect)

        tmp_index = cleaned_data[:, iFeature] == -999
        cleaned_data[tmp_index, iFeature] = tmp_mean

    tX = cleaned_data
"""

# feature ranking according to the correlation between feature and label
#damagedFeature = np.array([5, 12, 26, 27, 28, 23, 24, 25, 22, 29])

# correlation
cor = np.zeros(nFeature)
for iFeature in range(nFeature):
    tX_tmp, y_tmp  = data_cleaning(tX[:, iFeature], y)
    tmp = np.corrcoef(tX_tmp,y_tmp)**2 # correlation
    cor[iFeature] = tmp[1][0]

orderedPower = -np.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])


# call of the function to find the best correlation
print(orderedInd)
#print(plot_feature(tX,orderedInd,y))


# code Arthur
#for i in range(len(damagedFeature)):
#    orderedInd.remove(damagedFeature[i])



#allHistogram1Fig(orderedInd, tX, y)

oneHistogram(orderedInd, tX, y)

#lambda_ = np.logspace(-5,-2,20)
#degree = np.linspace(1,10,10)
#lambdaStudy(orderedInd, tX, y, lambda_, degree)
