#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import time
import sys
import glob
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import argh
import time

class GetSizeByColTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col):
        self.group_col = group_col

        # inititlizing vars
        self.cols_data_size_ = pd.Series([], dtype='int')
        self.cols_data_mean_ = pd.Series([], dtype='float64')

    def fit(self, X, y=None):
        cols_data_size_ = X.groupby(self.group_col).size()
        cols_data_mean_ = y.groupby(X[self.group_col]).mean()

        self.cols_data_size_ = self.cols_data_size_.append(cols_data_size_)
        self.cols_data_mean_ = self.cols_data_mean_.append(cols_data_mean_)
        return self

    def transform(self, X):
        out = pd.DataFrame()
        out['count'] = X[self.group_col].map(self.cols_data_size_).fillna(0)
        out['mean'] = X[self.group_col].map(self.cols_data_size_).fillna(-1)
        return out

    def get_feature_names(self):
        return ['count', 'mean']


class GetColsSumGroupByCol(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, sum_cols):
        self.group_col = group_col
        self.sum_cols = sum_cols

        self.cols_data_ = pd.DataFrame([], columns=self.sum_cols)

    def fit(self, X, y=None):
        cols_data_ = X.groupby(X[self.group_col])[self.sum_cols].sum()
        self.cols_data_ = self.cols_data_.append(cols_data_)
        return self

    def transform(self, X):
        out = pd.DataFrame()
        for col in self.sum_cols:
            out[col] = X[self.group_col].map(self.cols_data_[col])
        return out

    def get_feature_names(self):
        return self.sum_cols

class GetColsMeanGroupByCol(BaseEstimator, TransformerMixin):
    def __init__(self, group_col, sum_cols, fill_na=-1):
        self.group_col = group_col
        self.sum_cols = sum_cols
        self.fill_na = fill_na

        self.cols_data_ = pd.DataFrame([], columns=sum_cols)

    def fit(self, X, y=None):
        mul_table = X[self.sum_cols].mul(y, axis=0)
        mul_table = mul_table.replace(0.0, np.nan)
        cols_data_ = mul_table.groupby(X[self.group_col]).mean()
        self.cols_data_ = self.cols_data_.append(cols_data_)
        return self

    def transform(self, X):
        out = pd.DataFrame()
        for col in self.sum_cols:
            out[col] = X[self.group_col].map(self.cols_data_[col])

        if self.fill_na != None:
            out = out.fillna(self.fill_na)

        return out

    def get_feature_names(self):
        return self.sum_cols

class JoinTableTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, df_join, on, cols=None, drop_cols=None):
        self.on = on
        self.cols = cols
        self.drop_cols = drop_cols
        self.df_join = df_join

    def fit(self, X, y=None):
        self.out_cols_ = self.cols
        if self.out_cols_ == None:
            self.out_cols_ = self.df_join.columns
        elif self.out_cols_ == 'All':
            self.out_cols_ = [col for col in X.columns] + [col for col in self.df_join.columns]
        return self

    def transform(self, X):
        out = X.join(self.df_join, on=X[self.on])
        out = out[self.out_cols_]
        if self.drop_cols:
            out = out.drop(self.drop_cols, axis=1)
        return out
    
    def get_feature_names(self):
        return self.out_cols_


class PassthroughTransformer(BaseEstimator, TransformerMixin):
    """ Passes some of the columns unchanged"""
    def __init__(self, cols=[]):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.cols]
    
    def get_feature_names(self):
        return self.cols

def get_pipline():
    print ("Loading game data")
    df_games = pd.read_feather('processed/encoded_games_detailed_info.fethear')
    df_games = df_games.set_index('id')

    sum_and_means_cols = [a for a in df_games.columns]
    sum_and_means_cols.remove('review__count')
    sum_and_means_cols.remove('review__mean')

    features = FeatureUnion([
        ('game', PassthroughTransformer(['review__count', 'review__mean'] + sum_and_means_cols)),
        ('user_reviews', GetSizeByColTransformer('user')), 
        ('sum', GetColsSumGroupByCol('user', sum_and_means_cols)),
        ('mean', GetColsMeanGroupByCol('user', sum_and_means_cols)),
    ])

    pipeline = Pipeline([
        ('gamedata', JoinTableTransformer(df_games, on='game_id', cols='All')),
        ('features', features)
    ])
    return pipeline

def transform_data(is_train, input_f, output_f, pipeline):
    X = pd.read_feather(input_f)
    y = X['rating']
    X = X[['game_id', 'user']]
    
    if is_train:
        res = pipeline.fit_transform(X, y)
    else:
        res = pipeline.transform(X)
    
    new_df = pd.DataFrame(res, columns=pipeline.steps[-1][1].get_feature_names())
    new_df['rating'] = y
    new_df.to_feather(output_f)

def get_took_time(t):
    if t < 60:
        return ("{} Sec".format(t))
    elif t < 3600:
        return ("{} Mintes and {} Sec".format(  
            t // 60, t % 60
        ))
    else:
        return ("{} Hours, {} Mintes and {} Sec".format(  
            t // 3600, t % 3600 // 60, t % 60
        ))

def loop(pipeline, df_type, is_train):
    loop_start_time = int(time.time())
    for i, filename in enumerate( glob.glob(f'processed/{df_type}/*.feather') ):
        inner_time = int(time.time())
        basename = filename.split('/')[-1]
        print (f"Processing file #{i+1}: {basename}...", end =" ")
        target_filename = (f'processed/{df_type}/ready/{basename}')
        transform_data(is_train, filename, target_filename, pipeline)
        print ("took " + get_took_time(int(time.time()) - inner_time))
    print (f"ALL {df_type.upper()} TOOK " + get_took_time(int(time.time()) - loop_start_time))
        
    

def main():
    pipeline = get_pipline()
    start_time = time.time()
    print ("Training")
    loop(pipeline, 'train', True)
    print ("Validation")
    loop(pipeline, 'val', False)
    print ("Test")
    loop(pipeline, 'test', False)

    print ("ALL FILES TOOK " + get_took_time(int(time.time()) - start_time))
        

if __name__ == '__main__':
    main()