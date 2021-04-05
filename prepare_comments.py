#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import time
import sys
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
import argh



class GetSizeByColTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col):
        self.group_col = group_col

    def fit(self, X, y=None):
        self.cols_data_size_ = X.groupby(self.group_col).size()
        self.cols_data_mean_ = y.groupby(X[self.group_col]).mean()
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

    def fit(self, X, y=None):
        self.cols_data_ = X.groupby(X[self.group_col])[self.sum_cols].sum()
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

    def fit(self, X, y=None):
        mul_table = X[self.sum_cols].mul(y, axis=0)
        mul_table = mul_table.replace(0.0, np.nan)
        self.cols_data_ = mul_table.groupby(X[self.group_col]).mean()
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
    
    def get_feature_names():
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

def main(input_f, output_f):
    print ("Loading rating data")
    X = pd.read_feather(input_f)
    y = X['rating']
    X = X[['game_id', 'user']]
    
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

    res = pipeline.fit_transform(X, y)
    new_df = pd.DataFrame(res, columns=features.get_feature_names())
    new_df['rating'] = y
    new_df.to_feather(output_f)

if __name__ == '__main__':
   argh.dispatch_command(main) 
