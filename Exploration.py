from proj1_helpers import load_csv_data # how to import all ? . ?
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
import seaborn as sns # for visualization (heatmap)

import random

#TODO: documentation
def data_cleaning(iFeatureTrain, y):

    #exclude values = -999
    conserved_index_wash1 = iFeatureTrain != -999
    tX_wash1 = iFeatureTrain[conserved_index_wash1]

    #exclude abs(value) > 5*standard deviation
    tX_wash1_std = np.std(tX_wash1)
    conserved_index_wash2 = (np.abs(tX_wash1) <= tX_wash1_std * 5)

    tX_wash2 = tX_wash1[conserved_index_wash2]

    y_wash1 = y[conserved_index_wash1]
    y_wash2 = y_wash1[conserved_index_wash2]

    return tX_wash2, y_wash2
    #return tX_wash1, y_wash1


train_path = 'data/train.csv'
test_path = 'data/test.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)
y_te, tX_te, ids_te = load_csv_data(train_path, sub_sample=False)

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


"""
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
"""

# feature ranking according to the correlation between feature and label
damagedFeature = np.array([5, 12, 26, 27, 28, 23, 24, 25, 22, 29])
cor = np.zeros(nFeature)

## correlation when -999 are dropped
cor = np.zeros(nFeature)

for iFeature in range(nFeature):
    tX_tmp, y_tmp  = data_cleaning(tX[:, iFeature], y)
    # correlation
    tmp = np.corrcoef(tX_tmp,y_tmp)**2
    cor[iFeature] = tmp[1][0]

orderedPower = -np.sort(-cor)
orderedInd = sorted(range(nFeature), key=lambda k: -cor[k])
#print(orderedPower)
#print(orderedInd)

# code Arthur
#for i in range(len(damagedFeature)):
#    orderedInd.remove(damagedFeature[i])

## Build matrix with all features but with aberrant + missing observation removed
cleaned_data = tX
print(cleaned_data.shape)

for iFeature in range(nFeature):
    conserved_index_wash1 = cleaned_data[:, iFeature] != -999
    cleaned_data = cleaned_data[conserved_index_wash1,:]

print(cleaned_data.shape)


# for each feature, display the histogram with color = label (y)
damagedFeature0 = orderedInd

histogram = True
lambdaStudy = False

seed = 1
k_fold = 4

if histogram:
    plt.figure() # for the histogram

for i, iFeature in enumerate(damagedFeature0):

    tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
    tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

    # split data in k fold
    k_indices = build_k_indices(y_tmp, k_fold, seed)

    maximum = max(tX_tmp)
    minimum = min(tX_tmp)

    data1 = tX_tmp[y_tmp == -1]
    data2 = tX_tmp[y_tmp == 1]

    if histogram:
        # plot the figure
        bins = np.linspace(minimum, maximum, 80)

        plt.subplot(6,5,i+1)
        plt.hist(data1, bins, alpha=0.5, label='x')
        plt.hist(data2, bins, alpha=0.5, label='y')
        plt.xlim(minimum,maximum)
        #plt.legend(loc='upper right')
        plt.title('Feature n° ' + str(damagedFeature0[i]))


    if lambdaStudy:
        lambda_ = np.logspace(-5,-2,20)
        degree = np.linspace(1,10,10)

        degree_label = degree.astype(int)
        lambda_label = np.around(lambda_, 5)

        rmse_tr = np.zeros([degree.shape[0],lambda_.shape[0]])
        rmse_te = np.zeros([degree.shape[0], lambda_.shape[0]])

        best_parameter = [999, 0, 0]

        for i, degree_i in enumerate(degree):
            for j, lambda_i in enumerate(lambda_):

                rmse_tr_tmp = []
                rmse_te_tmp = []

                # cross-validation
                for k in range(k_fold):
                    loss_tr, loss_te = cross_validation(y_tmp, tX_tmp, k_indices, k, lambda_, degree)
                    rmse_tr_tmp.append(loss_tr)
                    rmse_te_tmp.append(loss_te)

                rmse_tr[i,j] = np.mean(rmse_tr_tmp)
                tmp = np.mean(rmse_te_tmp)
                rmse_te[i,j] = tmp

                if tmp < best_parameter[0]:
                    best_parameter = [tmp, degree_i, lambda_i]

        print(best_parameter)

        # test set
        #tx_tr = build_poly(tX_tmp, best_parameter[1])
        #weights, mse = ridge_regression(y_tmp, tx_tr, best_parameter[2])
        #tx_te = build_poly(tX_te_tmp, best_parameter[1])
        #y_pred = predict_labels(weights, tx_te)
        #name = 'submission-1.csv'
        #create_csv_submission(ids, y_pred, name)

        diff_rmse_tr = rmse_tr - rmse_tr.min()
        #mse_losses_norm = (mse_losses - mse_losses.mean())/mse_losses.std()
        #mse_losses_scal = (mse_losses - mse_losses.min()) / (mse_losses.max() - mse_losses.min())
        plt.figure()
        ax = sns.heatmap(diff_rmse_tr,vmin=0,vmax=0.03,annot=True,xticklabels=lambda_label, yticklabels=degree_label, cmap="YlGnBu")
        plt.show()

        diff_rmse_te = rmse_te - rmse_te.min()
        plt.figure()
        ax = sns.heatmap(diff_rmse_te, vmin=0,vmax=0.03, annot=True, xticklabels=lambda_label,yticklabels=degree_label, cmap="YlGnBu")
        plt.show()


        #plt.figure()
        #plt.plot(degree, mse_losses)
        #plt.title('mseLoss in function of Lambda')
        #plt.show()

if histogram:
    plt.savefig('histogram.png') # for the histogram
    plt.show() # for the histogram
