import numpy as np


def standardize(x):
    """Standardize the data set"""
    ctr = x - np.mean(x, axis=0)
    stdi = ctr / np.std(ctr, axis=0)
    return stdi


def data_cleaning(iFeatureTrain, y):

    #exclude values = -999
    conserved_index_wash1 = iFeatureTrain != -999
    tX_wash1 = iFeatureTrain[conserved_index_wash1]
    #index_wash1 = iFeatureTrain == -999
    #iFeatureTrain[index_wash1] = np.median(iFeatureTrain)
    #tX_wash1 = iFeatureTrain
    #exclude abs(value) > 5*standard deviation
    #tX_wash1_std = np.std(tX_wash1)
    #conserved_index_wash2 = (np.abs(tX_wash1) <= tX_wash1_std * 5)
    #index_wash2 = (np.abs(tX_wash1) >= tX_wash1_std * 5)

    #tX_wash2 = tX_wash1[conserved_index_wash2]
    #tX_wash1[index_wash2] = np.median(iFeatureTrain)

    y_wash1 = y[conserved_index_wash1]
    #y_wash1= y
    #y_wash2 = y_wash1[conserved_index_wash2]

    #return tX_wash2, y_wash2
    return tX_wash1, y_wash1

"""
def data_cleaning(iFeatureTrain, y): # fonction Arthur

    #exclude values = -999
    #conserved_index_wash1 = iFeatureTrain != -999
    #tX_wash1 = iFeatureTrain[conserved_index_wash1]
    index_wash1 = iFeatureTrain == -999
    iFeatureTrain[index_wash1] = np.median(iFeatureTrain)
    tX_wash1 = iFeatureTrain
    #exclude abs(value) > 5*standard deviation
    tX_wash1_std = np.std(tX_wash1)
    #conserved_index_wash2 = (np.abs(tX_wash1) <= tX_wash1_std * 5)
    index_wash2 = (np.abs(tX_wash1) >= tX_wash1_std * 5)

    #tX_wash2 = tX_wash1[conserved_index_wash2]
    tX_wash1[index_wash2] = np.median(iFeatureTrain)

    #y_wash1 = y[conserved_index_wash1]
    y_wash1= y
    #y_wash2 = y_wash1[conserved_index_wash2]

    #return tX_wash2, y_wash2
    return tX_wash1, y_wash1
"""