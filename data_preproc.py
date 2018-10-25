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

def preprocessing(tX_feat22,y_feat22):

    # remove uninteresting features
    uninterestingFeature = [15, 18, 20, 22, 25, 28]
    tX_feat22 = np.delete(tX_feat22, uninterestingFeature, 1)

    print(tX_feat22.shape)

    #nSample, nFeature = tX_feat22.shape

    # remove features full of -999
    m = np.mean(tX_feat22, axis=0)
    uninterestingFeature_index = np.where(m == -999)
    tX_feat22 = np.delete(tX_feat22, uninterestingFeature_index, 1)

    print(tX_feat22.shape)

    tX_feat22,y_feat22 = data_cleaning(tX_feat22, y_feat22, imputation = True, outlier_removal = True)
    return tX_feat22,y_feat22

