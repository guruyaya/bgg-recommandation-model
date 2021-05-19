# bgg-recommandation-model
This project was created as a final project for the Data Science Course in Naya Collage. It's designed to handle the data that could be found here
(https://www.kaggle.com/jvanelteren/boardgamegeek-reviews)

## How to run
This process is designed to work on linux systems
0. pip install -r requirements.txt
1. download the data from kaggle, and storing the CSV files in a folder called data
2. Create a folder named processed
3. Create a folder named models
4. ./nlp_preprocess_desc.py data/games_detailed_info.csv
5. ./run_starting_from_nlp.sh

This should end up creating a working xgboost model in your models folder

Note that the run on a 4G 2CPU amazon instance took around 1 hour. 

### Using the model
An example of model usage can be followed on this jupyter notebook here:
(https://github.com/guruyaya/bgg-recommandation-model/blob/main/exploring_model.ipynb)

### The model in the wild
I cannot promise anything at this point, but for now you can find an app using this model in this address
(http://bgg.inspect-element.net/)
