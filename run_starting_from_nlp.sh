#!/bin/bash
mv processed/nlp_game_details.feather .
find processed/ -type f -exec rm {} \;
mv nlp_game_details.feather processed/

python shuffle_15m_comments.py data/bgg-15m-reviews.csv
python prepare_game_df.py processed/nlp_game_details.feather
python create_user_grouped_files.py 200000 500000 500000
python prepare_comments.py

python xdg_inc.py processed