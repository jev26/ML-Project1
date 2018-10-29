import numpy as np


def standardize(x):
    """Standardize the data set"""
    ctr = x - np.mean(x, axis=0)
    stdi = ctr / np.std(ctr, axis=0)
    return stdi


def data_cleaning(tX, y, imputation = True, outlier_removal = True):
    """Clean the data set"""

    tX_return = tX
    y_return = y

    nSample, nFeature = tX.shape

    for iFeature in range(nFeature):

        iFeatureTrain = tX[:, iFeature]

        #index_for_mean = iFeatureTrain != -999
        #new_value = np.median(iFeatureTrain[index_for_mean])

        unique_value, index_value = np.unique(iFeatureTrain, return_inverse=True)
        highest_freq_index = np.argmax(np.bincount(index_value))
        new_value = unique_value[highest_freq_index]

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

def preprocessing(complete_tX,complete_y,complete_ids):
    """Split the data set regarding to the value of feature PRI_jet_num"""
    for Nbrjet in range(3):

        if Nbrjet == 2:
            tX_feat22 = complete_tX[complete_tX[:,22] > 1]
            y_feat22 = complete_y[complete_tX[:,22] > 1]
            ids_feat22 = complete_ids[complete_tX[:,22] > 1]

        else:
            tX_feat22 = complete_tX[complete_tX[:,22] == Nbrjet]
            y_feat22 = complete_y[complete_tX[:,22] == Nbrjet]
            ids_feat22 = complete_ids[complete_tX[:,22] == Nbrjet]

        # remove uninteresting features
        uninterestingFeature = [15, 18, 20, 22, 25, 28]
        tX_feat22 = np.delete(tX_feat22, uninterestingFeature, 1)

        # remove features full of -999
        m = np.mean(tX_feat22, axis=0)
        uninterestingFeature_index = np.where(m == -999)
        tX_feat22 = np.delete(tX_feat22, uninterestingFeature_index, 1)

        tX_feat22,y_feat22 = data_cleaning(tX_feat22, y_feat22, imputation = True, outlier_removal = True)

        # split train and test set
        if Nbrjet == 0:

            model0 = dict(
                tX_tr = tX_feat22[y_feat22 != 0],
                y_tr = y_feat22[y_feat22 != 0],
                tX_te = tX_feat22[y_feat22 == 0],
                te_id= ids_feat22[y_feat22 == 0]
            )
        elif Nbrjet == 1:

            model1 = dict(
                tX_tr = tX_feat22[y_feat22 != 0],
                y_tr = y_feat22[y_feat22 != 0],
                tX_te = tX_feat22[y_feat22 == 0],
                te_id= ids_feat22[y_feat22 == 0]
            )
        else :

            model2 = dict(
                tX_tr = tX_feat22[y_feat22 != 0],
                y_tr = y_feat22[y_feat22 != 0],
                tX_te = tX_feat22[y_feat22 == 0],
                te_id= ids_feat22[y_feat22 == 0]
            )

    return model0, model1, model2

