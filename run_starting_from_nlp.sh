#!/bin/bash
mv processed/nlp_game_details.feather .
find processed/ -type f -exec rm {} \;
mv nlp_game_details.feather processed/

python shuffle_15m_comments.py data/bgg-15m-reviews.csv
python prepare_game_df.py processed/nlp_game_details.feather
python prepare_comments.py

./prepare_all_user_files.sh
python xdg_inc.py processed
