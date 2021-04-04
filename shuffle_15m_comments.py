#!/use/bin/env python
import pandas as pd
df = pd.read_csv('data/bgg-15m-reviews.csv.gz', usecols=['user', 'ID', 'rating'])
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

print ("Number of reviews:", len(df))	

# df.to_feather('processed/train/bgg-reviews.feather', index=True)
