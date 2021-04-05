#!/use/bin/env python
import pandas as pd
import argh
def main(filename):
    """ name the reviews filename (can handle zipped and gziped file, to get shuffles feather version in processed folder"
    """
    df = pd.read_csv(filename, usecols=['user', 'ID', 'rating'])
    print ("Loaded")
    df = df[ ~df['user'].isna() ]
    print ("Dropped empty user")
    print( df.head() )
    df = df.rename(columns={'ID': 'game_id'})
    print ("Renamed")
    print (df.head())
    df = df.reindex(columns=['user', 'game_id', 'rating'])
    print ("Reordered")
    print (df.head())
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print ("Reordered")
    print (df.head())

    num_reviews = len(df)
    print ("Number of reviews:", num_reviews)	

    max_train = num_reviews * 6 // 10
    max_val = num_reviews * 8 // 10
    print ("Splitting")
    df.iloc[:max_train].to_feather('processed/bgg-reviews_train.feather')
    df.iloc[max_train:max_val].reset_index().to_feather('processed/bgg-reviews_val.feather')
    df.iloc[max_val:].reset_index().to_feather('processed/bgg-reviews_test.feather')
    print ("Done")

if __name__ == '__main__':
    argh.dispatch_command(main)
