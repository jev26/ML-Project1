from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
import seaborn as sns # for visualization (heatmap)

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



def data_cleaning(iFeatureTrain, y):

    #exclude values = -999
    #conserved_index_wash1 = iFeatureTrain != -999
    #tX_wash1 = iFeatureTrain[conserved_index_wash1]
    index_wash1 = iFeatureTrain == -999
    iFeatureTrain[index_wash1] = np.median(iFeatureTrain)
    tX_wash1 = iFeatureTrain
    #exclude abs(value) > 5*standard deviation
    tX_wash1_std = np.std(tX_wash1)
    #conserved_index_wash2 = (np.abs(tX_wash1) <= tX_wash1_std * 5)
    index_wash2 = (np.abs(tX_wash1) >= tX_wash1_std * 5)

    #tX_wash2 = tX_wash1[conserved_index_wash2]
    tX_wash1[index_wash2] = np.median(iFeatureTrain)

    #y_wash1 = y[conserved_index_wash1]
    y_wash1= y
    #y_wash2 = y_wash1[conserved_index_wash2]

    #return tX_wash2, y_wash2
    return tX_wash1, y_wash1

# count the number of 'value' per feature
def RatioCountValue(value,tX):
    # value is typically -999 or 0
    nSample, nFeature = tX.shape
    for i in range(nFeature):
        tmp = sum(tX[:, i] == value)
        print('Feature n° ' + str(i) + " " + str(tmp))
        print('Ratio ' + str(tmp/nSample))

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

    print(cleaned_data.shape)
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


# feature ranking according to the correlation between feature and label
damagedFeature = np.array([5, 12, 26, 27, 28, 23, 24, 25, 22, 29])
cor = np.zeros(nFeature)

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


# code Arthur
#for i in range(len(damagedFeature)):
#    orderedInd.remove(damagedFeature[i])


# for each feature, display the histogram with color = label (y)
damagedFeature0 = orderedInd

histogram_sameFig = False # for each feature (on one figure), display the histogram with color = label (y)
histogram_diffFig = True # for each feature, display the histogram with color = label (y)
lambdaStudy = False # for each feature, display a heatmap showing the test error depending on lambda and degree

seed = 1
k_fold = 4

if histogram_sameFig:
    plt.figure() # for the histogram

for i, iFeature in enumerate(damagedFeature0):

    iFeature = 22
    tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
    #tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

    # split data in k fold
    k_indices = build_k_indices(y_tmp, k_fold, seed)

    maximum = max(tX_tmp)
    minimum = min(tX_tmp)

    data1 = tX_tmp[y_tmp == -1]
    data2 = tX_tmp[y_tmp == 1]

    if histogram_sameFig or histogram_diffFig:
        # plot the figure
        bins = np.linspace(minimum, maximum, 80)

        if histogram_sameFig:
            plt.subplot(6,5,i+1)
        elif histogram_diffFig:
            plt.figure()

        plt.hist(data1, bins, alpha=0.5, label='x')
        plt.hist(data2, bins, alpha=0.5, label='y')
        plt.xlim(minimum,maximum)
        plt.title('Feature n° ' + str(damagedFeature0[i]))

        if histogram_diffFig:
            plt.legend(loc='upper right')
            plt.show()

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
        diff_rmse_te = rmse_te - rmse_te.min()
        plt.figure()
        ax = sns.heatmap(diff_rmse_te, vmin=0,vmax=0.03, annot=True, xticklabels=lambda_label,yticklabels=degree_label, cmap="YlGnBu")
        plt.show()

        # on test set
        #tx_tr = build_poly(tX_tmp, best_parameter[1])
        #weights, mse = ridge_regression(y_tmp, tx_tr, best_parameter[2])
        #tx_te = build_poly(tX_te_tmp, best_parameter[1])
        #y_pred = predict_labels(weights, tx_te)
        #name = 'submission-1.csv'
        #create_csv_submission(ids, y_pred, name)

        #plt.figure()
        #plt.plot(degree, mse_losses)
        #plt.title('mseLoss in function of Lambda')
        #plt.show()

if histogram_sameFig:
    plt.savefig('histogram.png') # for the histogram
    plt.show() # for the histogram




