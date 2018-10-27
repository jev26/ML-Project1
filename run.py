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

print('Proprocessing of data')

model0, model1, model2 = preprocessing(complete_tX,complete_y,complete_ids)

all_model = [model0, model1, model2]

#best_param = [1.67683293681101e-07,2.2229964825261955e-07,1.2648552168552957e-07]
degree = 2
lambda_ = np.logspace(-10, 1, 20)

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

    print('Generating features')
    tX_newfeat = generate_features(model_i['tX_tr'], degree)
    print('Shape of new features', tX_newfeat.shape)

    print('start learning')
    best_parameter, losses_tr, losses_te,best_loss, std_te, accuracy_mean, accuracy_std = learning(tX_newfeat, model_i['y_tr'], degree,lambda_) #ou model_i['best_param']
    model_i.update({'best_param': best_parameter})
    model_i.update({'losses_tr': losses_tr})
    model_i.update({'losses_te': losses_te})
    model_i.update({'best_loss': best_loss})
    model_i.update({'std_te': std_te})
    model_i.update({'accuracy_mean': accuracy_mean})
    model_i.update({'accuracy_std': accuracy_std})
    #print('learning done')

    print('Starting ridge regression')
    w,_ = ridge_regression(model_i['y_tr'], tX_newfeat, best_parameter)
    #w,_ = ridge_regression(model_i['y_tr'], tX_newfeat, best_param[i])

    tX_te_newfeat = generate_features(model_i['tX_te'], degree)
    pred = predict_labels(w, tX_te_newfeat)

    ids_final = np.append(ids_final, model_i['te_id'])
    y_final = np.append(y_final, pred)

print('Shape of output: ', y_final.shape)
print('Creating submission')

create_csv_submission(ids_final, y_final, "final_submission.csv")

# print best parameter for each model
plt.figure()
for i, model_i in enumerate(all_model):
    plt.subplot(3, 1, i + 1)
    print('Best param: ', model_i['best_param'])
    plotError(model_i['losses_tr'], model_i['losses_te'], lambda_)
    print('losses_te ', model_i['best_loss'], ' +/- ', model_i['std_te'])
    print('accuracy ', model_i['accuracy_mean'], ' +/- ', model_i['accuracy_std'])
plt.show()
