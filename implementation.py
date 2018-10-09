import numpy as np
import matplotlib.pyplot as plt


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
    # ***************************************************
    error = y - tx.dot(weights)
    grad = -(1 / len(error)) * tx.T.dot(error)
    return grad, e



def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    batch_size = 1
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad, e = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_mse(e)
            w = w - gamma * grad
    return loss, w


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






