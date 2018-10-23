import numpy as np
import itertools
from data_preproc import *


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
        print(b)
        PF[:, a] = X[:, b].prod(1)

    return PF

def log_feature(tX):
    for i in range(np.size(tX),2):
        tX[:,i] = np.log(tX[:,i] - min(tX[:,i]) + 1)
    return tX

def feature_log_feature(tX):
    for i in range(np.size(tX),2):
        tX = tX*log_feature(tX)
    return tX

def tanh_feature(tX):
    tX = np.tanh(tX)
    return tX

def feature_tanh_feature(tX):
    tX = tX*tanh_feature((tX))
    return tX

def cos_feature(tX):
    tX = np.cos(tX)
    return tX

def feature_cos_feature(tX):
    tX = tX*cos_feature(tX)
    return tX

def exp_one_over_feature(tX):
    tX = np.exp(1/(tX-np.min(tX)+1))
    return tX




def generate_features(tX,orderedInd,y):
    new_tX = np.empty([tX.shape[0], 1])
    for i in range(len(orderedInd)):
        tX_tmp, y_tmp = data_cleaning((tX[:,orderedInd[i]]), y, False, True)
        if i == 0:
            new_tX = tX_tmp
        else:
            print((new_tX.shape, tX_tmp.shape))
            new_tX = np.column_stack((new_tX, tX_tmp))

    log_tX = log_feature(tX)
    tanh_tX = tanh_feature(tX)
    cos_tX = cos_feature(tX)
    sin_tX = exp_one_over_feature(tX)

    tx_array = np.array([log_tX, tanh_tX, cos_tX, sin_tX])

    for a in tx_array:
        new_tX = np.column_stack((new_tX, a))


    tX_final = polynomial_features(new_tX, 2)

    return tX_final