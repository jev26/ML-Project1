from crossval import *
from features_func import *

def learning(tX, y, lambda_):
    """Function used to find the best parameters of a model"""
    losses_te = []
    std_te = []
    losses_tr = []
    accuracy_mean = []
    accuracy_std = []

    #tX_newfeat = generate_features(tX, degree)
    #print('Shape of generated features: ', tX_newfeat.shape)

    seed = 1
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_parameter = 0
    best_loss = 999

    print('tX inside CV',tX.shape)

    for j, lambda_i in enumerate(lambda_):
        #print('Current lambda tested: ', lambda_i)

        rmse_tr_tmp = []
        rmse_te_tmp = []
        accuracy_tmp = []

        # cross-validation
        for k in range(k_fold):
            #print('Pass number ', k)
            loss_tr, loss_te, score = cross_validation(y, tX, k_indices, k, lambda_i)

            rmse_te_tmp.append(loss_te)
            rmse_tr_tmp.append(loss_tr)
            accuracy_tmp.append(score)


        tmp = np.mean(rmse_te_tmp)

        losses_te.append(tmp)
        #std_te.append(np.std(rmse_te_tmp))
        losses_tr.append(np.mean(rmse_tr_tmp))
        #accuracy_mean.append(np.mean(accuracy_tmp))
        #accuracy_std.append(np.std(accuracy_tmp))

        if tmp < best_loss:
            best_loss = tmp
            best_parameter = lambda_i
            std_te = np.std(rmse_te_tmp)
            accuracy_mean = np.mean(accuracy_tmp)
            accuracy_std = np.std(accuracy_tmp)


    return best_parameter, losses_tr, losses_te, best_loss, std_te, accuracy_mean, accuracy_std

