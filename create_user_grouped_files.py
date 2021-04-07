#!/usr/bin/env python
import pandas as pd
import numpy as np
import sys

def split_file(file_name, target_lib, lines=200_000):
    print ("Starting")
    user_df = pd.read_feather(file_name)
    user_df = user_df[ ~user_df['user'].isna() ]
    user_df['user'] = user_df['user'].str.strip()
    print ("Loaded")
    user_df = user_df.sort_values(['user']).reset_index()
    print ("Sorted")
    max_loc_per_user = user_df.groupby('user').apply(lambda df: df.index.max())
    print ("Grouped")

    df_from = 0
    while True:
        ref_num = df_from + lines
        df_to = max_loc_per_user[ max_loc_per_user >= ref_num].min()

        if df_to == np.nan:
            save_df = user_df.loc[df_from:]
        else:
            save_df = user_df.loc[df_from:df_to]

        if len(save_df) == 0:
            break
        
        print (f"Saving from {df_from} to {df_to}")
        print ("starting user: {}, ending user: {}".format(save_df.iloc[0].user, save_df.iloc[-1].user))
        r = save_df.reset_index(drop=True)
        try:
            r.to_feather(f'{target_lib}/reviews_from_{df_from:08d}_to_{df_to:08d}.feather')
        except ValueError:
            max_df = int(save_df.index.max())
            r.to_feather(f'{target_lib}/reviews_from_{df_from:08d}_to_{max_df:08d}.feather')
        df_from = df_to + 1
    print ("Done")

if __name__ == '__main__':
    print ("Train")
    split_file('processed/bgg-reviews_train.feather','processed/train', 10_000)
    print ("Val")
    split_file('processed/bgg-reviews_val.feather','processed/val', 50_000)
    print ("Test")
    split_file('processed/bgg-reviews_test.feather','processed/test', 50_000)
