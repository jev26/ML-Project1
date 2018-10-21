import numpy as np
import itertools


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))  # create a vector with the x inputs
    for deg in range(1, degree + 1):  # for each degree from 1 to the degree chosen(inclusive)
        poly = np.c_[poly, np.power(x, deg)]  # concatenate!
    return poly

def polynomial_features(X, degree):
    """polynomial feature function that create a new features matrix with all combinations
    of features with degree less than or equal to the degree"""
    #get the number of samples and features from the X matrix
    nb_samples, nb_features = X.shape

    #create an iterator that lets us iterate over all combinations of numbers from 0 to nb_features-1
    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))

    #use that iterator to get the total number of features of the output
    nb_output = sum(1 for _ in combi)

    #initiate an empty array for the output
    PF = np.empty([nb_samples, nb_output])

    #instantiate the iterator again
    combi = itertools.chain.from_iterable(
        itertools.combinations_with_replacement(range(nb_features), i) for i in range(degree + 1))

    #create the polynomial features by iterating and multipliying the columns
    for a, b in enumerate(combi):
        PF[:, a] = X[:, b].prod(1)

    return PF