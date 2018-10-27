from crossval import *
from features_func import *

def learning(tX, y, degree, lambda_):
    losses_te = []
    losses_tr = []
    tX_newfeat = generate_features(tX, degree)
    print('Shape of generated features: ', tX_newfeat.shape)

    seed = 1
    k_fold = 4

    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)

    best_parameter = 0
    best_loss = 999

    for j, lambda_i in enumerate(lambda_):
        print('Current lanbda tested: ', lambda_i)

        rmse_tr_tmp = []
        rmse_te_tmp = []

        # cross-validation
        for k in range(k_fold):
            print('Pass number ', k)
            loss_tr, loss_te,_ = cross_validation(y, tX_newfeat, k_indices, k, lambda_i)
            rmse_te_tmp.append(loss_te)
            rmse_tr_tmp.append(loss_tr)

        tmp = np.mean(rmse_te_tmp)
        tmp2 = np.mean(rmse_tr_tmp)

        losses_te.append(tmp)
        losses_tr.append(tmp2)

        if tmp < best_loss:
            best_loss = tmp
            best_parameter = lambda_i

    return best_parameter, losses_tr, loss_te

