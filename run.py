from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from data_preproc import preprocessing
from features_func import generate_features
from Learning import learning

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)
nSample_tr, nFeature_tr = tX.shape

test_path = 'data/test.csv'
y_te, tX_te, ids_te = load_csv_data(test_path, sub_sample=False)
nSample_te, nFeature_te = tX.shape

whole_tX = np.vstack((tX,tX_te))

model_0, model_1, model_2 = learning(tX, y)




#oneHistogram(range(0, nFeature), tX_feat22, y_feat22)

