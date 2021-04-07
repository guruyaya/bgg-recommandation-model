import xgboost as xgb
import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error as MSE
import sys
import time

train_files = glob(sys.argv[1] + '/train/ready/*.feather')
val_files = glob(sys.argv[1] + '/val/ready/*.feather')

batch_size = len(train_files)
val_size = len(val_files)



def validate(model):
    y_pr = np.array([])
    y_true = np.array([])
    importance = np.array([])

    for val_file in val_files:
        X_val = pd.read_feather(val_file)
        y_true = np.concatenate([y_true, X_val['rating'].to_numpy()])
        X_val = X_val.drop('rating', axis=1)
        importance = np.concatenate([importance, np.log( X_val['user_reviews__count'] + 0.001 )])

        y_pr = np.concatenate([y_pr, model.predict(xgb.DMatrix(X_val))])
    return MSE(y_true, y_pr, squared=False, sample_weight=importance)

def iteration (i, model):
    global loss_history, best_model, best_model_loss
    # train data
    train_file = train_files[i % batch_size]
    print (train_file)
    X = pd.read_feather(train_file)
    y = X['rating']
    importance = np.log( X['user_reviews__count'] + 0.001 )
    X = X.drop('rating', axis=1)

    model = xgb.train({'learning_rate': 0.07,'process_type': 'default',
                      'refresh_leaf': True, 'reg_alpha': 3}, 
                      dtrain=xgb.DMatrix(X, y, weight=importance), xgb_model = model)

    loss = validate(model)
    if loss < best_model_loss:
        best_model = model
        best_model.save_model('models/model_data.json')

    loss_history.to_csv('models/loss_history.csv')

    print('MSE itr@{}: {}'.format(i, loss))
    loss_history = loss_history.append([{'iter':i, 'filenum': i % batch_size, 'batch': i // batch_size, 'MSE': loss}])
    if len(loss_history) > 3:
        print ([loss > loss_history.iloc[-num]['MSE'] for num in range(2, 5, 1)])
        if all([loss > loss_history.iloc[-num]['MSE'] for num in range(2, 5, 1)]):
            print ("last 3 losses did not go smaller, break")
            return True, model
    return False, model


iterations = 3
model = None
loss_history = pd.DataFrame([],columns=['iter', 'filenum', 'batch', 'MSE'])
best_model = None
best_model_loss = 999999

if __name__ == '__main__':
    start_time = int(time.time())
    for i in range(iterations * batch_size):
        iter_start_time = int(time.time())
        is_done, model = iteration(i, model)
        total_time = int(time.time()) - start_time
        iter_time = int(time.time()) - iter_start_time
        print ("Iter {} took {} hours, {} minutes and {} seconds ({}:{} passed)".format(
            i, iter_time // 3600, iter_time % 3600 // 60, iter_time % 60,
            total_time // 3600, total_time % 3600 // 60
        ))
        if is_done:
            break
    total_time = int(time.time()) - start_time
    print ("Finshed all in {} hours, {} minutes and {} seconds".format(
        total_time // 3600, total_time % 3600 // 60, total_time % 60
    ))

