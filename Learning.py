from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from data_preproc import data_cleaning

train_path = 'data/train.csv'
test_path = 'data/test.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)
y_te, tX_te, ids_te = load_csv_data(test_path, sub_sample=False)

# print(y.shape) => (250000,)
# print(tX.shape) => (250000, 30) => 30 features
# print(ids.shape) => (250000,)

nSample, nFeature = tX.shape
nSample_te, nFeature_te = tX_te.shape

"""
data1 = tX[y == -1]
data2 = tX[y == 1]

print(data1.shape) # (164333, 30)
print(data2.shape) # (85667, 30)
"""

# correlation
cor = np.zeros(nFeature)
for iFeature in range(nFeature):
    tX_tmp, y_tmp  = data_cleaning(tX[:, iFeature], y)
    tmp = np.corrcoef(tX_tmp,y_tmp)**2 # correlation
    cor[iFeature] = tmp[1][0]

orderedPower = -np.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])

# call of the function to find the best correlation
#print(orderedInd)
#print(plot_feature(tX,orderedInd,y))


lambda_ = np.logspace(-5,-2,20)
degree = np.linspace(1,10,10)
#lambdaStudy(orderedInd, tX, y, lambda_, degree)

seed = 1
k_fold = 4

for i, iFeature in enumerate(orderedInd):
    tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
    # tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

    #if tX_tmp.size != 0:

    # split data in k fold
    k_indices = build_k_indices(y_tmp, k_fold, seed)

    maximum = max(tX_tmp)
    minimum = min(tX_tmp)

    #degree_label = degree.astype(int)
    #lambda_label = np.around(lambda_, 5)

    rmse_tr = np.zeros([degree.shape[0],lambda_.shape[0]])
    rmse_te = np.zeros([degree.shape[0], lambda_.shape[0]])

    best_parameter = [999, 0, 0]

    for i, degree_i in enumerate(degree):
        for j, lambda_i in enumerate(lambda_):

            rmse_tr_tmp = []
            rmse_te_tmp = []

            # cross-validation
            for k in range(k_fold):
                loss_tr, loss_te = cross_validation(y_tmp, tX_tmp, k_indices, k, lambda_i, degree_i.astype(int))
                rmse_tr_tmp.append(loss_tr)
                rmse_te_tmp.append(loss_te)

            rmse_tr[i,j] = np.mean(rmse_tr_tmp)
            tmp = np.mean(rmse_te_tmp)
            rmse_te[i,j] = tmp

            if tmp < best_parameter[0]:
                best_parameter = [tmp, degree_i, lambda_i]

    print(best_parameter)

    # heatmap for test error (test set from CV)
    #diff_rmse_te = rmse_te - rmse_te.min()
    #plt.figure()
    #ax = sns.heatmap(diff_rmse_te, vmin=0,vmax=0.03, annot=True, xticklabels=lambda_label,yticklabels=degree_label, cmap="YlGnBu")
    #plt.show()


