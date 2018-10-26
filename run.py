from proj1_helpers import *
import matplotlib.pyplot as plt
import numpy as np
from implementation import *
from Visualization import *
from data_preproc import preprocessing
from features_func import generate_features
from Learning import learning

train_path = 'data/train.csv'
y, tX, ids = load_csv_data(train_path, sub_sample=False)
nSample_tr, nFeature_tr = tX.shape

test_path = 'data/test.csv'
_, tX_te, ids_te = load_csv_data(test_path, sub_sample=False)
y_te = np.zeros((tX_te.shape[0],))
nSample_te, nFeature_te = tX.shape

complete_tX = np.vstack((tX,tX_te))
complete_y = np.append(y,y_te)
complete_ids = np.append(ids,ids_te)

model0, model1, model2 = preprocessing(complete_tX,complete_y,complete_ids)

all_model = [model0, model1, model2]

y_final = []
ids_final = []

for model_i in all_model:
    print(model_i['tX_tr'].shape)
    print(model_i['y_tr'].shape)
    print(model_i['tX_te'].shape)
    print(model_i['te_id'].shape)

    nSample, nFeature = model_i['tX_tr'].shape
    #oneHistogram(range(0, nFeature), model_i['tX_tr'], model_i['y_tr'])

    print('start learning')
    best_parameter = learning(model_i['tX_tr'], model_i['y_tr']) #ou model_i['best_param']
    model_i.update({'best_param': best_parameter})
    print('learning done')
    w,_ = ridge_regression(model_i['y_tr'], model_i['tX_tr'], best_parameter[2])
    print('ridge regression')

    pred = predict_labels(w, model_i['tX_te'])
    ids_final = np.append(ids_final, model_i['te_id'])
    y_final = np.append(y_final, pred)

print('creating submission')

create_csv_submission(ids_final, y_final, "final_submission.csv")

# print best parameter for each model
for model_i in all_model:
    print(model_i['best_param'])
