import numpy as np
import itertools
from data_preproc import *
from personal_helpers import sigmoid


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

def tanh_feature(tX):
    tX = np.tanh(tX)
    return tX

def cos_feature(tX):
    tX = np.cos(tX)
    return tX

def sigmoid_feature(tX):
    tX = sigmoid((tX-np.mean(tX,axis=0)))
    return tX




def generate_features(tX, degree):
    log_tX = log_feature(tX)
    tanh_tX = tanh_feature(tX)
    cos_tX = cos_feature(tX)
    sig_tX = sigmoid_feature(tX)

    tx_array = np.array([log_tX, tanh_tX, cos_tX, sig_tX])

    for a in tx_array:
        tX = np.column_stack((tX, a))

    tX_final = polynomial_features(tX, degree)

    return tX_final