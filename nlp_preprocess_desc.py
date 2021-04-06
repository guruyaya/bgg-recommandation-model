import argh
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import nltk

def identify_tokens(review):
    tokens = nltk.word_tokenize(review)
    # taken only words (not punctuation)
    token_words = [w for w in tokens if w.isalpha()]
    return token_words

def stem_list(my_list):
    stemming = PorterStemmer()
    stemmed_list = [stemming.stem(word) for word in my_list]
    return stemmed_list

def rejoin_words(my_list):
    joined_words = ( " ".join(my_list))
    return joined_words

def preprocess_desc(X):
    df = X.copy()
    description = df['description'].fillna('')
    description = description.apply(identify_tokens)
    description = description.apply(stem_list)
    df['processed'] = description.apply(rejoin_words)
    df.drop('description', axis=1)
    return df

def main(filename):
    df = pd.read_csv(filename)
    df = preprocess_desc(df)
    df.to_feather('processed/nlp_game_details.feather')

if __name__ == '__main__':
    argh.dispatch_command(main)
