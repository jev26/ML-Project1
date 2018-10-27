from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from data_preproc import preprocessing
from features_func import *

def learning(tX, y, degree, lambda_):

    tX_newfeat = generate_features(tX, degree)
    print(tX_newfeat.shape)

    seed = 1
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_parameter = 0
    best_loss = 999

    for j, lambda_i in enumerate(lambda_):
        print(lambda_i)

        #rmse_tr_tmp = []
        rmse_te_tmp = []

        # cross-validation
        for k in range(k_fold):
            _, loss_te,_ = cross_validation(y, tX_newfeat, k_indices, k, lambda_i)
            rmse_te_tmp.append(loss_te)

        tmp = np.mean(rmse_te_tmp)

        if tmp < best_loss:
            best_loss = tmp
            best_parameter = lambda_i

    return best_parameter

