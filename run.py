from Visualization import *
from data_preproc import preprocessing
from features_func import generate_features

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

print('Proprocessing of data')

model0, model1, model2 = preprocessing(complete_tX,complete_y,complete_ids)

all_model = [model0, model1, model2]

best_param = [1.584893192461114e-07,2.5118864315095823e-07,5.179474679231202e-09]
degree = 2

y_final = []
ids_final = []

scores = []

for i, model_i in enumerate(all_model):
    print('Shape of tX training: ', model_i['tX_tr'].shape)
    print('Shape of y training: ', model_i['y_tr'].shape)
    print('Shape of tX test: ', model_i['tX_te'].shape)
    print('Shape of te ids: ', model_i['te_id'].shape)

    nSample, nFeature = model_i['tX_tr'].shape
    #oneHistogram(range(0, nFeature), model_i['tX_tr'], model_i['y_tr'])

    #print('start learning')
    #best_parameter = learning(model_i['tX_tr'], model_i['y_tr'], degree) #ou model_i['best_param']
    #model_i.update({'best_param': best_parameter})
    #print('learning done')

    print('Generating features')
    tX_newfeat = generate_features(model_i['tX_tr'], degree)
    print('Starting ridge regression')
    w,_ = ridge_regression(model_i['y_tr'], tX_newfeat, best_param[i])

    tX_te_newfeat = generate_features(model_i['tX_te'], degree)

    pred = predict_labels(w, tX_te_newfeat)

    ids_final = np.append(ids_final, model_i['te_id'])
    y_final = np.append(y_final, pred)

print('Shape of output: ', y_final.shape)
print('Creating submission')

create_csv_submission(ids_final, y_final, "final_submission.csv")

# print best parameter for each model
#for model_i in all_model:
#    print(model_i['best_param'])
