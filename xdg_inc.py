import xgboost as xgb
import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import mean_squared_error as MSE
import sys
import time

train_files = glob('processed/train/ready/*.feather')
val_files = glob('processed/val/ready/*.feather')

batch_size = len(train_files)
val_size = len(val_files)


class ModelTrainer:
    def __init__(self, iterations=10, compare_last=3, starting_model_folder=None):
        self.iterations = iterations
        self.compare_last = compare_last

        self.model = None
        self.loss_history = pd.DataFrame([],columns=['iter', 'filenum', 'batch', 'MSE'])
        self.is_done = False
        self.best_model = None
        self.best_model_loss = 999999

    def validate(self):
        y_pr = np.array([])
        y_true = np.array([])
        importance = np.array([])

        for val_file in val_files:
            X_val = pd.read_feather(val_file)
            y_true = np.concatenate([y_true, X_val['rating'].to_numpy()])
            X_val = X_val.drop('rating', axis=1)
            importance = np.concatenate([importance, 
                np.log( X_val['user_reviews__count'] + 0.001 )])

            y_pr = np.concatenate([y_pr, self.model.predict(xgb.DMatrix(X_val))])
        return MSE(y_true, y_pr, squared=False, sample_weight=importance)

    def train_model (self, train_file):
        X = pd.read_feather(train_file)
        y = X['rating']
        importance = np.log( X['user_reviews__count'] + 0.001 )
        X = X.drop('rating', axis=1)

        self.model = xgb.train({'learning_rate': 0.07,'process_type': 'default',
                        'refresh_leaf': True, 'reg_alpha': 3}, 
                        dtrain=xgb.DMatrix(X, y, weight=importance), xgb_model = self.model)
        
    def iteration (self, i):
        # train data
        train_file = train_files[i % batch_size]
        print (train_file)
        self.train_model(train_file)
        loss = self.validate()

        if loss < self.best_model_loss:
            self.best_model = self.model
            self.best_model.save_model('models/model_data.json')

        print('MSE itr@{}: {}'.format(i, loss))
        self.loss_history = self.loss_history.append([{'iter':i, 'filenum': i % batch_size, 'batch': i // batch_size, 'MSE': loss}])
        self.loss_history.to_csv('models/loss_history.csv')
        if len(self.loss_history) > 3:
            last_losses = ([loss > self.loss_history.iloc[-num]['MSE'] 
                for num in range(2, 2+self.compare_last, 1)])
            print (last_losses)
            if all(last_losses):
                print ("last {self.compare_last} losses did not go smaller, break")
                self.is_done = True

iterations = 10

if __name__ == '__main__':
    start_time = int(time.time())
    model_trainer = ModelTrainer()

    for i in range(iterations * batch_size):
        iter_start_time = int(time.time())
        model_trainer.iteration(i)
        total_time = int(time.time()) - start_time
        iter_time = int(time.time()) - iter_start_time
        print ("Iter {} took {} hours, {} minutes and {} seconds ({}:{} passed)".format(
            i, iter_time // 3600, iter_time % 3600 // 60, iter_time % 60,
            total_time // 3600, total_time % 3600 // 60
        ))
        if model_trainer.is_done:
            break
    total_time = int(time.time()) - start_time
    print ("Finshed all in {} hours, {} minutes and {} seconds".format(
        total_time // 3600, total_time % 3600 // 60, total_time % 60
    ))

