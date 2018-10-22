# study of the 22 feature
from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
import seaborn as sns # for visualization (heatmap)

import random

def data_cleaning(iFeatureTrain, y):

    #exclude values = -999
    conserved_index_wash1 = iFeatureTrain != -999
    tX_wash1 = iFeatureTrain[conserved_index_wash1]
    #index_wash1 = iFeatureTrain == -999
    #iFeatureTrain[index_wash1] = np.median(iFeatureTrain)
    #tX_wash1 = iFeatureTrain
    #exclude abs(value) > 5*standard deviation
    #tX_wash1_std = np.std(tX_wash1)
    #conserved_index_wash2 = (np.abs(tX_wash1) <= tX_wash1_std * 5)
    #index_wash2 = (np.abs(tX_wash1) >= tX_wash1_std * 5)

    #tX_wash2 = tX_wash1[conserved_index_wash2]
    #tX_wash1[index_wash2] = np.median(iFeatureTrain)

    y_wash1 = y[conserved_index_wash1]
    #y_wash1= y
    #y_wash2 = y_wash1[conserved_index_wash2]

    #return tX_wash2, y_wash2
    return tX_wash1, y_wash1

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)

nSample, nFeature = tX.shape

#data_0 = tX[tX[:,22]==0] # (99913, 30)
#data_1 = tX[tX[:,22]==1] # (77544, 30)
#data_2 = tX[tX[:,22]==2] # (50379, 30)
#data_3 = tX[tX[:,22]==3] # (22164, 30)

# correlation
for feat22 in range(4):

    print(feat22)

    tX_feat22 = tX[tX[:,22]==feat22]
    y_feat22 = y[tX[:,22]==feat22]
    y_tmp = y_feat22

    cor = np.zeros(nFeature)
    for iFeature in range(nFeature):
        tX_tmp = tX_feat22[:, iFeature]
        #tX_tmp, y_tmp  = data_cleaning(tX[:, iFeature], y)
        tmp = np.corrcoef(tX_tmp,y_tmp)**2 # correlation
        cor[iFeature] = tmp[1][0]

    orderedPower = -np.sort(-cor)
    orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])

    print(orderedInd)

    for i, iFeature in enumerate(orderedInd):

        #tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
        #tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)
        tX_tmp, y_tmp2 = data_cleaning(tX_feat22[:, iFeature], y_tmp)

        print(tX_tmp.shape)
        print(y_tmp2.shape)

        if tX_tmp.size != 0:

            maximum = max(tX_tmp)
            minimum = min(tX_tmp)

            data1 = tX_tmp[y_tmp2 == -1]
            data2 = tX_tmp[y_tmp2 == 1]

            # plot the figure
            bins = np.linspace(minimum, maximum, 80)

            plt.figure()

            plt.hist(data1, bins, alpha=0.5, label='x')
            plt.hist(data2, bins, alpha=0.5, label='y')
            plt.xlim(minimum,maximum)
            plt.title('Feature n° ' + str(orderedInd[i]))

            plt.legend(loc='upper right')
            plt.show()
