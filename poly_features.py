import numpy as np
import itertools


def polynomial_features(X, degree):
    nb_samples, nb_features = X.shape

    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))
    nb_output = sum(1 for _ in combi)

    PF = np.empty([nb_samples, nb_output])

    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))

    for a, b in enumerate(combi):
        PF[:, a] = X[:, b].prod(1)

    return PF