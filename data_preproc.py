import numpy as np


def standardize(x):
    """Standardize the data set"""
    ctr = x - np.mean(x, axis=0)
    stdi = ctr / np.std(ctr, axis=0)
    return stdi


def normalize(x, high=1.0, low=0.0):
    mins = np.min(x, axis=0)
    maxs = np.max(x, axis=0)
    rng = maxs - mins
    return high - (((high - low) * (maxs - x)) / rng)

"""
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

def data_cleaning(tX, y, imputation = True, outlier_removal = True):

    tX_return = tX
    y_return = y

    nSample, nFeature = tX.shape

    for iFeature in range(nFeature):

        iFeatureTrain = tX[:, iFeature]

        index_for_mean = iFeatureTrain != -999
        new_value = np.median(iFeatureTrain[index_for_mean])

        if imputation:
            index_to_replace = iFeatureTrain == -999
            tX[index_to_replace, iFeature] = new_value
            #tX_return[index_to_replace] = new_value

        iFeatureTrain = tX[:, iFeature]

        if outlier_removal: # exclude abs(value) > 5 * standard deviation
            outlier_threshold = np.std(iFeatureTrain)
            to_replace_index = (np.abs(iFeatureTrain) > 5 * outlier_threshold)

            tX_return[to_replace_index, iFeature] = new_value

    return tX_return, y_return

