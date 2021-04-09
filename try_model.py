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

test_files = glob.glob('processed/test/ready/*.feather')
for test_file in test_files:
    print (test_file)
    X_test = pd.read_feather(test_file)
    y_true = np.concatenate([y_true, X_test['rating'].to_numpy()])
    X_test = X_test.drop('rating', axis=1)
    importance = np.concatenate([importance, np.log( X_test['user_reviews__count'] + 0.001 )])

    y_pr = np.concatenate([y_pr, model.predict(xgb.DMatrix(X_test))])

loss = MSE(y_true, y_pr, squared=False, sample_weight=importance)

print (f"Loss is {loss}")
