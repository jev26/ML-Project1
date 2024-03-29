from proj1_helpers import *
from implementation import *

def cross_validation(y, x, k_indices, k, lambda_):
    """return the loss of ridge regression."""

    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    x_te = x[te_indice]
    y_te = y[te_indice]

    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)
    x_tr = x[tr_indice]
    y_tr = y[tr_indice]

    score_table = []

    # ridge regression
    #print('Ridge inside cross val started')
    weights, loss = ridge_regression(y_tr, x_tr, lambda_)

    y_te_predict = predict_labels(weights, x_te)
    score = (y_te_predict == y_te).mean()
    score_table.append(score)

    # calculate the loss for train and test data
    e_tr = y_tr - x_tr.dot(weights)
    loss_tr = np.sqrt(2 * compute_mse(e_tr))

    e_te = y_te - x_te.dot(weights)
    loss_te = np.sqrt(2 * compute_mse(e_te))

    return loss_tr, loss_te, np.array(score_table)


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)