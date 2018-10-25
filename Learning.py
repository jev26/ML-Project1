from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from data_preproc import preprocessing
from features_func import generate_features

def learning(tX, y):

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

        tX_feat22 = generate_features(tX_feat22, 2)
        print(tX_feat22.shape)

        lambda_ = np.logspace(-5, -2, 20)
        degree = np.linspace(1, 10, 10)

        seed = 1
        k_fold = 4

        # split data in k fold
        k_indices = build_k_indices(y_feat22, k_fold, seed)

        best_parameter = [999, 0, 0]

        for i, degree_i in enumerate(degree):
            for j, lambda_i in enumerate(lambda_):

                #rmse_tr_tmp = []
                rmse_te_tmp = []

                # cross-validation
                for k in range(k_fold):
                    _, loss_te = cross_validation(y_feat22, tX_feat22, k_indices, k, lambda_i, degree_i.astype(int))
                    rmse_te_tmp.append(loss_te)

                tmp = np.mean(rmse_te_tmp)

                if tmp < best_parameter[0]:
                    best_parameter = [tmp, degree_i, lambda_i]

        if Nbrjet == 0:
            model_0 = best_parameter
        elif Nbrjet == 1:
            model_1 = best_parameter
        else:
            model_2 = best_parameter

    return model_0, model_1, model_2

