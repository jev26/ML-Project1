from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)

for feat22 in range(3):

    print(feat22)

    if feat22 == 2:
        tX_feat22 = tX[tX[:,22] > 1]
        y_feat22 = y[tX[:,22] > 1]

    else:
        tX_feat22 = tX[tX[:,22] == feat22]
        y_feat22 = y[tX[:,22] == feat22]

    # remove uninteresting features
    uninterestingFeature = [15, 18, 20, 22, 25, 28]
    tX_feat22 = np.delete(tX_feat22, uninterestingFeature, 1)

    print(tX_feat22.shape)

    #nSample, nFeature = tX_feat22.shape

    # remove features full of -999
    m = np.mean(tX_feat22, axis=0)
    uninterestingFeature_index = np.where(m == -999)
    tX_feat22 = np.delete(tX_feat22, uninterestingFeature_index, 1)

    nSample, nFeature = tX_feat22.shape
    print(tX_feat22.shape)

    tX_feat22,y_feat22 = data_cleaning(tX_feat22, y_feat22, imputation = True, outlier_removal = True)
    oneHistogram(range(0, nFeature), tX_feat22, y_feat22)


    #oneHistogram(np.linspace(0, nFeature), tX_feat22, y_feat22)
    #data_cleaning(iFeatureTrain, y, NaN_removal=True, imputation=False, outlier_removal=False)

