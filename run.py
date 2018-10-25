from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from preprocessing import preprocessing

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)

test_path = 'data/test.csv'
y_te, tX_te, ids_te = load_csv_data(test_path, sub_sample=False)

#whole_tX = np.concatenate(tX,tX_te)
#print(whole_tX.shape)

for Nbrjet in range(3):

    print(Nbrjet)

    if Nbrjet == 2:
        tX_feat22 = tX[tX[:,22] > 1]
        y_feat22 = y[tX[:,22] > 1]

    else:
        tX_feat22 = tX[tX[:,22] == Nbrjet]
        y_feat22 = y[tX[:,22] == Nbrjet]

    tX_feat22, y_feat22 = preprocessing(tX_feat22, y_feat22)
    nSample, nFeature = tX_feat22.shape

    oneHistogram(range(0, nFeature), tX_feat22, y_feat22)
