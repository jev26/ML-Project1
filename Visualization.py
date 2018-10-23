from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
import seaborn as sns # for visualization (heatmap)
from data_preproc import data_cleaning


def allHistogram1Fig(damagedFeature0, tX, y): # for each feature (on one figure), display the histogram with color = label (y)
    plt.figure()  # for the histogram
    for i, iFeature in enumerate(damagedFeature0):
        tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
        # tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

        if tX_tmp.size != 0:
            maximum = max(tX_tmp)
            minimum = min(tX_tmp)

            data1 = tX_tmp[y_tmp == -1]
            data2 = tX_tmp[y_tmp == 1]

            # plot the figure
            bins = np.linspace(minimum, maximum, 80)
            plt.subplot(6, 5, i + 1)
            plt.hist(data1, bins, alpha=0.5, label='x')
            plt.hist(data2, bins, alpha=0.5, label='y')
            plt.xlim(minimum, maximum)
            plt.title('Feature n° ' + str(damagedFeature0[i]))

    #plt.savefig('histogram.png')  # for the histogram
    plt.show()  # for the histogram


def oneHistogram(damagedFeature0, tX, y): # for each feature, display the histogram with color = label (y)
    seed = 1
    k_fold = 4

    for i, iFeature in enumerate(damagedFeature0):
        tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
        # tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

        if tX_tmp.size != 0:

            maximum = max(tX_tmp)
            minimum = min(tX_tmp)

            data1 = tX_tmp[y_tmp == -1]
            data2 = tX_tmp[y_tmp == 1]

            # plot the figure
            bins = np.linspace(minimum, maximum, 80)
            plt.figure()
            plt.hist(data1, bins, alpha=0.5, label='x')
            plt.hist(data2, bins, alpha=0.5, label='y')
            plt.xlim(minimum, maximum)
            plt.title('Feature n° ' + str(damagedFeature0[i]))
            plt.legend(loc='upper right')
            plt.show()



def lambdaStudy(damagedFeature0, tX, y, lambda_, degree): # for each feature, display a heatmap showing the test error depending on lambda and degree
    seed = 1
    k_fold = 4

    for _, iFeature in enumerate(damagedFeature0):
        tX_tmp, y_tmp = data_cleaning(tX[:, iFeature], y)
        # tX_te_tmp, y_te_tmp = data_cleaning(tX_te[:, iFeature], y_te)

        #if tX_tmp.size != 0:

        # split data in k fold
        k_indices = build_k_indices(y_tmp, k_fold, seed)

        maximum = max(tX_tmp)
        minimum = min(tX_tmp)

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
        plt.title('Feature n° ' + str(iFeature))
        plt.show()

