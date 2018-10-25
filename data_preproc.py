import numpy as np


def standardize(x):
    """Standardize the data set"""
    ctr = x - np.mean(x, axis=0)
    stdi = ctr / np.std(ctr, axis=0)
    return stdi


def normalize(x, high=100.0, low=0.0):
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - x)) / rng)


def data_cleaning(iFeatureTrain, y, NaN_removal = True, imputation = False, outlier_removal = False):

    tX_return = iFeatureTrain
    y_return = y

    if NaN_removal: # exclude values == -999
        conserved_index = iFeatureTrain != -999
        tX_return = iFeatureTrain[conserved_index]

        y_return = y[conserved_index]

    elif imputation:
        index_for_mean = iFeatureTrain != -999
        new_value = np.median(iFeatureTrain[index_for_mean])

        index_to_replace = iFeatureTrain == -999
        tX_return[index_to_replace] = new_value

    if outlier_removal: # exclude abs(value) > 5 * standard deviation
        outlier_threshold = np.std(tX_return)
        conserved_index = (np.abs(tX_return) <= 5 * outlier_threshold)

        tX_return = tX_return[conserved_index]
        y_return = y_return[conserved_index]

    return tX_return, y_return

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