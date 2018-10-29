import numpy as np

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
    weights = initial_w

    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            error = minibatch_y - minibatch_tx.dot(weights)
            grad = -(1 / len(error)) * minibatch_tx.T.dot(error)
            mse_loss = compute_mse(error)
            weights = weights - gamma * grad

    return weights, mse_loss

def ridge_regression(y, tx, lambda_):
    part1 = 2 * tx.shape[0] * lambda_ * np.eye(tx.shape[1])
    a = tx.T.dot(tx) + part1
    b = tx.T.dot(y)
    weights = np.linalg.solve(a, b)

    error = y - tx @ weights
    rmse_loss = np.sqrt(compute_mse(error)*2)

    return weights, rmse_loss


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    weights = initial_w
    Losses = []
    threshold = 1e-7

    for iter in range(max_iters):
        # calculate Loss
        tmp = sigmoid(tx.dot(weights))
        Loss = -((1 - y).T.dot(np.log(1 - tmp)) + y.T.dot(np.log(tmp)))
        # calculate the gradient
        Grad = tx.T.dot(tmp - y)
        # update by gradient descend method
        weights = weights - gamma * Grad
        # condition to stop iteration if we already converged. Gain us computation time
        Losses.append(Loss)
        if len(Losses) > 1 and np.abs(Losses[-1] - Losses[-2]) < threshold:
            break

    return weights, Loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    weights = initial_w
    Losses = []
    threshold = 1e-7

    for iter in range(max_iters):
        # calculate Loss
        tmp = sigmoid(tx.dot(weights))
        Loss = -((1 - y).T.dot(np.log(1 - tmp)) + y.T.dot(np.log(tmp))) + (lambda_ / 2) * weights.T.dot(weights)
        # calculate the gradient
        Grad = tx.T.dot(tmp - y) + lambda_ * weights
        # update by gradient descent method
        weights = weights - gamma * Grad
        # condition to stop iteration if we already converged. Gain us computation time
        Losses.append(Loss)
        if len(Losses) > 1 and np.abs(Losses[-1] - Losses[-2]) < threshold:
            break

    return weights, Loss


""""personal helpers"""

def sigmoid(x):
    """Sigmoid function"""
    return 1/(1+np.exp(-x))


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



def compute_mse(error):
    """Returns the MSE"""
    return (1 / (2 * np.size(error))) * np.sum(error * error)

