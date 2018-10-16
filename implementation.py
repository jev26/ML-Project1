import numpy as np
import matplotlib.pyplot as plt

def scale_linear_bycolumn(rawpoints, high=1.0, low=0.0):
    mins = np.min(rawpoints, axis=0)
    maxs = np.max(rawpoints, axis=0)
    tmp = maxs - mins
    return high - (((high - low) * (maxs - rawpoints)) / tmp)
def compute_mse(error):
    return (1 / (2 * np.size(error))) * np.sum(error * error)

def least_squares(y, tx):
    # print(tx.shape)
    # print(y.shape)
    b = np.transpose(tx) @ y
    a = np.transpose(tx) @ tx
    weights = np.linalg.solve(a, b)

    error = y - tx @ weights
    mse_loss = compute_mse(error)
    return weights, mse_loss


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    weights = initial_w
    for n_iter in range(max_iters):
        error = y - tx.dot(weights)
        grad = -(1 / len(error)) * tx.T.dot(error)

        mse_loss = compute_mse(error)
        weights = weights - gamma * grad

    return weights, mse_loss


def compute_stoch_gradient(y, tx, weights):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    error = y - tx.dot(weights)
    grad = -(1 / len(error)) * tx.T.dot(error)
    return grad, error



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    """Stochastic gradient descent algorithm."""
    weights = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad, error = compute_stoch_gradient(minibatch_y, minibatch_tx, weights)
            mse_loss = compute_mse(error)
            weights = weights - gamma * grad
    return weights, mse_loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    part1 = 2*tx.shape[0]*lambda_*np.eye(tx.shape[1])

    a = tx.T.dot(tx) + part1
    b = tx.T.dot(y)
    weights = np.linalg.solve(a,b)
    error = y - tx.dot(weights)
    mse_loss = compute_mse(error)
    return weights, mse_loss


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]





def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # ***************************************************
    polynomial = np.ones((len(x),1))
    # format de ones pour avoir une matrice et pas un array. concatenate veut une matrice
    xpower = np.zeros((len(x),degree))
    for i in range (1, degree+1) :
        xpower[:,i-1] = np.power(x,i)
    polynomial = np.concatenate((polynomial,xpower),axis=1)
    return polynomial

