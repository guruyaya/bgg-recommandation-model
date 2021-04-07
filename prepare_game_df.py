#!/usr/bin/env python3
"""
This file contains the process to handle the full game info
"""
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import KBinsDiscretizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk
import pandas as pd
import numpy as np
import pickle
import sys
import argh

# preparing nltk downloads

class Cat2BOW(BaseEstimator, TransformerMixin):
    def __init__(self, col, min_df=0):
        self.col = col
        self.min_df = min_df
        
    def fit(self, X, y=None):
        adjusted_col = self.prepare_for_bow( X[self.col] )
        self.fitted_vectorizer_ = CountVectorizer(min_df=self.min_df).fit(adjusted_col)

        return self

    def transform(self, X):
        adjusted_col = self.prepare_for_bow( X[self.col] )
        return self.fitted_vectorizer_.transform(adjusted_col).toarray()
        
    @staticmethod
    def prepare_for_bow(col):
        return (col.
            str.replace(' ', '_').
            str.replace(r"['\"],_['\"]", ' ', regex=True).
            str.replace(r"^\[['\"]", '', regex=True).
            str.replace(r"['\"]\]$", '', regex=True).
            str.lower().
            str.replace(r"[^a-z0-9 ]", '_', regex=True).
            fillna('')
        )

    def get_feature_names(self):
        return self.fitted_vectorizer_.get_feature_names()

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

class GrpupingEncoder(TransformerMixin, BaseEstimator):
    
    def __init__(self, col='', groups=[], prefix = '',
            override_labels = None, group_name_margin = 1,
            first_group_format=None, last_group_format=None,
            ranges_format = '{}-{}'):
            
        self.col = col
        self.override_labels = override_labels
        self.groups = groups
        self.prefix = prefix
        self.first_group_format = first_group_format
        self.last_group_format = last_group_format
        self.group_name_margin = group_name_margin
        self.ranges_format = '{from}-{to}'

    def set_labels(self):
        labels = [self.ranges_format.format(**{'from':self.groups[i]+self.group_name_margin, 'to':self.groups[i+1]}) 
            for i in range(len(self.groups)-1)]
        labels[0] = self.first_group_format.format(**{'from': self.groups[0]+self.group_name_margin, 'to': self.groups[1]})
        labels[-1] = self.last_group_format.format(**{'from': self.groups[-2]+self.group_name_margin, 'to': self.groups[-1]})
        return labels

    def fit(self, X, y=None):
        print ("GrpupingEncoder fit")
        self.labels = self.override_labels
        if self.labels == None:
            self.labels = self.set_labels()

        return self
    
    def transform(self, X):
        print ("GrpupingEncoder transform")
        data_labled = pd.cut(X[self.col], self.groups, labels=self.labels)
        data_labled.index = X.index

        ret_df = pd.DataFrame([[] for i in range(len( X ))], index=X.index)
        for label in self.labels:
            ret_df[label] = np.where(data_labled==label, 1, 0)

        return ret_df.to_numpy()

    def get_feature_names(self):
        return self.labels

class DescTokeniser(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=100, min_df=0.05, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
    
    def fit(self, X, y=None):
        self.vectorizer_.fit(df['processed'])
        return self

    def transform(self, X):
        return self.vectorizer_.transform(df['processed']).toarray()

    def fit_transform(self, X, y=None):
        self.vectorizer_ = TfidfVectorizer(min_df=self.min_df,max_df=self.max_df,
            max_features=self.max_features,stop_words=stopwords.words("english"))
        return self.vectorizer_.fit_transform(X['processed']).toarray()

    def get_feature_names(self):
        return self.vectorizer_.get_feature_names()

class PlayerNumAnalyzer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        print ("PlayerNumAnalyzer fit")
        self.types = ['solo', 'couples', 'multiplayers', 'party']
        return self

    def transform(self, X):
        print ("PlayerNumAnalyzer predict")
        out = pd.DataFrame([])

        # out['solo'] = X['minplayers'] > 1
        out['solo'] = (X['minplayers'] == 1).astype('int')
        out['couples'] = ( 
            (X['minplayers'] >= 2) | 
            (   (X['minplayers'] == 0) & 
                ( (X['maxplayers'] >= 2) | (X['maxplayers'] == 0) )
            )
        ).astype('int')
        
        out['multiplayers'] = ( (X['maxplayers'] > 2) | (X['maxplayers'] == 0)).astype('int')
        out['party'] = ( (X['maxplayers'] >= 5) | (X['maxplayers'] == 0) ).astype('int')
        return out

    def get_feature_names(self):
        return self.types

class GameRatingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, reviews_file='processed/bgg-reviews_train.feather', file_type='feather'):
        self.reviews_file = reviews_file
        self.file_type = file_type

    @staticmethod
    def extract_data(df):
        out = {}
        out['count'] = df.size
        out['mean'] = df['rating'].mean()
        return pd.Series(out)

    def fit(self, X, y=None):
        if self.file_type == 'feather':
            data_df = pd.read_feather(self.reviews_file)
        elif self.file_type == 'csv':
            data_df = pd.read_feather(self.reviews_file)
        else:
            raise Exception(f'file type {self.file_type} is not supported')
        print ("Games data loaded")
        self.games_data_ = data_df.groupby('game_id').apply(
            self.extract_data
        )
        self.all_games_mean_ = data_df['rating'].mean()
        return self
        
    def transform(self, X):
        print ("Game transform")
        out = X[[]].join(self.games_data_)
        out['count'] = out['count'].fillna(0)
        out['mean'] = out['mean'].fillna(self.all_games_mean_)
        return out

    def get_feature_names(self):
        return ['count','mean',]

def main(filename):
    print("Starting Now!\n\n\n")
    nltk.download('punkt')
    nltk.download('stopwords')

    df = pd.read_feather(filename)

    years = (-10000, 1900, 1980, 1990, 2000, 2010, 2015, 21000)

    final_df = FeatureUnion([
        ('', PassthroughTransformer(['id',])),
        ('review', GameRatingTransformer()),
        ('years', GrpupingEncoder(col='yearpublished', groups=years, 
            first_group_format='Pre-{to}', last_group_format='Post-{from}')),
        ('min_age', GrpupingEncoder(col='minage', groups=(0, 3, 10, 12, 18, 100), 
            first_group_format='none', last_group_format='adults')),
        ('weight', GrpupingEncoder(col='averageweight', groups=np.arange(0, 5, 0.5), 
            first_group_format='less_than_{to:.2f}', last_group_format='more_than_{from:.2f}', group_name_margin=0.01, 
            ranges_format='{from:.2f}-{to:.2f}')),
        ('players', PlayerNumAnalyzer()),
        ('desc_nlp', DescTokeniser(max_features=200)),
        ('category', Cat2BOW('boardgamecategory', min_df=0.01)),
        ('mechanic', Cat2BOW('boardgamemechanic', min_df=0.01)),
    ])

    new_df = pd.DataFrame(final_df.fit_transform(df), 
                    columns=final_df.get_feature_names())
    print(new_df.shape)

    replacers = dict( [(name, name.replace('__','')) for name in new_df.columns if name.startswith('__')] )
    new_df.rename(columns=replacers, inplace=True)

    print(new_df.columns.str.split('__').str[0].value_counts())
    print (new_df.iloc[0])

    has_na = new_df.isna().sum(axis=0)
    print (has_na[has_na > 1])

    new_df.to_feather('processed/encoded_games_detailed_info.fethear')
    with open('models/game_df_pipline.pickle', 'wb') as f:
        pickle.dump(final_df, f) 

if __name__ == '__main__':
    argh.dispatch_command(main)
