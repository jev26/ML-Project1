# study of the 22 feature
from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *

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

    #allHistogram1Fig(orderedInd, tX_feat22, y_feat22)

    #oneHistogram(orderedInd, tX_feat22, y_feat22)

    lambda_ = np.logspace(-5,-2,20)
    degree = np.linspace(1,10,10)
    lambdaStudy(orderedInd, tX_feat22, y_feat22, lambda_, degree)
