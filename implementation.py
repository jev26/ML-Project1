import numpy as np
from proj1_helpers import *
import matplotlib.pyplot as plt

def least_squares(y, tx):
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
    part1 = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])

    a = tx.T.dot(tx) + part1
    b = tx.T.dot(y)
    weights = np.linalg.solve(a, b)
    return weights


