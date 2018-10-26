from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from data_preproc import preprocessing
from features_func import generate_features

def learning(tX, y):

    tX_newfeat = generate_features(tX, 1)
    print(tX_newfeat.shape)

    lambda_ = np.logspace(-5, -2, 20) #TODO set?
    degree = np.linspace(1, 10, 10) #TODO remove

    seed = 1
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_parameter = [999, 0, 0]

    for i, degree_i in enumerate(degree): #TODO remove
        for j, lambda_i in enumerate(lambda_):

            #rmse_tr_tmp = []
            rmse_te_tmp = []

            # cross-validation
            for k in range(k_fold):
                _, loss_te = cross_validation(y, tX_newfeat, k_indices, k, lambda_i)
                rmse_te_tmp.append(loss_te)

            tmp = np.mean(rmse_te_tmp)

            if tmp < best_parameter[0]:
                best_parameter = [tmp, degree_i, lambda_i]

    return best_parameter

