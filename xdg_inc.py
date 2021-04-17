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
        self.loss_history = pd.DataFrame([],columns=['iter', 'filenum', 'batch', 
            'MSE', 'MSE_no_importance', 'MSE_no_reviews', 'MSE_1_review', 'MSE_over_2_reviews', 'MSE_over_5_reviews', 'MSE_over_10_reviews', 'MSE_over_20_reviews'])
        self.is_done = False
        self.best_model = None
        self.best_model_loss = 999999

    def calc_importance(self, X):
        return np.log( (X['user_reviews__count']).apply(lambda n: max(n, 5)) + 1 );

    def validate(self, iternum, filenum, batchnum):
        y_pr = np.array([])
        y_true = np.array([])
        importance = np.array([])
        review_count = np.array([])

        for val_file in val_files:
            X_val = pd.read_feather(val_file)
            y_true = np.concatenate([y_true, X_val['rating'].to_numpy()])
            X_val = X_val.drop('rating', axis=1)
            importance = np.concatenate([importance, self.calc_importance(X_val)])
            review_count = np.concatenate([ review_count, X_val['user_reviews__count'] ])

            y_pr = np.concatenate([y_pr, self.model.predict(xgb.DMatrix(X_val))])
        out = pd.Series({
            'iter': iternum, 'filenum': filenum, 'batch': batchnum, 
            'MSE': MSE(y_true, y_pr, squared=False, sample_weight=importance), 
            'MSE_no_importance': MSE(y_true, y_pr, squared=False)
        })

        stats_list = [  ('MSE_no_reviews', review_count == 0), 
                        ('MSE_1_review', review_count == 1), 
                        ('MSE_over_2_reviews', review_count >= 2), 
                        ('MSE_over_5_reviews', review_count >= 5), 
                        ('MSE_over_10_reviews', review_count >= 10), 
                        ('MSE_over_20_reviews', review_count >=20) ]

        for stat_name, stat_mask in stats_list:
            try:
                out[stat_name] = MSE(y_true[stat_mask], y_pr[stat_mask], squared=False)
            except:
                print (f"Failed on {stat_name}")
                raise

        return out

    def train_model (self, train_file):
        X = pd.read_feather(train_file)
        y = X['rating']
        importance = self.calc_importance(X)
        X = X.drop('rating', axis=1)

        self.model = xgb.train({'learning_rate': 0.04,'process_type': 'default',
                        'refresh_leaf': True, 'reg_alpha': 1}, 
                        dtrain=xgb.DMatrix(X, y, weight=importance), xgb_model = self.model)
        
    def iteration (self, i):
        # train data
        train_file = train_files[i % batch_size]
        print (train_file)
        self.train_model(train_file)
        loss = self.validate(i, i % batch_size, i // batch_size)

        loss_examined = 'MSE_over_5_reviews'
        if loss[loss_examined] < self.best_model_loss:
            self.best_model = self.model
            self.best_model.save_model('models/model_data.json')

        print('MSE itr@{}: total:{:.4f}, no_weights: {:.4f}, no_review: {:.4f}, 1 review: {:.4f}, over 2: {:.4f}, over 5: {:.4f}, over 10: {:.4f}, over 20: {:.4f}'.format(
            *loss[ ['iter', 'MSE', 'MSE_no_importance', 'MSE_no_reviews', 'MSE_1_review', 'MSE_over_2_reviews', 
                'MSE_over_5_reviews', 'MSE_over_10_reviews', 'MSE_over_20_reviews'] ].to_list())
        )
        self.loss_history = self.loss_history.append(loss.to_dict(), ignore_index=True)
        self.loss_history.to_csv('models/loss_history.csv')
        if len(self.loss_history) > 3:
            last_losses = (self.loss_history.iloc[-(2+self.compare_last):-2][loss_examined])
            if all(last_losses < loss[loss_examined]):
                print (f"last {self.compare_last} losses did not go smaller, break")
                self.is_done = True

    def train_empty_file(self):
         train_file = 'processed/train/ready-empty.feather'
         print ("Traning on empty before starting the real traning")
         self.train_model(train_file)

iterations = 10

if __name__ == '__main__':
    start_time = int(time.time())
    model_trainer = ModelTrainer()

    model_trainer.train_empty_file()
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

