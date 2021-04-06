print ("Started")
import xgboost as xgb
import numpy as np
import pandas as pd
import glob 
import sys
from sklearn.metrics import mean_squared_error as MSE

print ("Imoprted")
model = xgb.Booster()
model.load_model(fname=sys.argv[1])
print ("Loaded")
print (model)

y_pr = np.array([])
y_true = np.array([])
importance = np.array([])

val_files = glob.glob('processed/val/ready/*.feather')
for val_file in val_files:
    X_val = pd.read_feather(val_file)
    y_true = np.concatenate([y_true, X_val['rating'].to_numpy()])
    X_val = X_val.drop('rating', axis=1)
    importance = np.concatenate([importance, np.log( X_val['user_reviews__count'] + 0.001 )])

    y_pr = np.concatenate([y_pr, model.predict(xgb.DMatrix(X_val))])

loss = MSE(y_true, y_pr, squared=False, sample_weight=importance)
print (f"Loss is {loss}")