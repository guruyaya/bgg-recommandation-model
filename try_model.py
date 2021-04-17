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
X_num_rev = np.array([])

test_files = glob.glob('processed/test/ready/*.feather')
for test_file in test_files:
    print (test_file)
    X_test = pd.read_feather(test_file)
    y_true = np.concatenate([y_true, X_test['rating'].to_numpy()])
    X_test = X_test.drop('rating', axis=1)

    X_num_rev = np.concatenate([X_num_rev, X_test['user_reviews__count']])
    importance = np.concatenate([importance, np.log( (X_test['user_reviews__count'] + 1) )])

    y_pr = np.concatenate([y_pr, model.predict(xgb.DMatrix(X_test))])

loss = pd.Series()
loss['Weighted'] = MSE(y_true, y_pr, squared=False, sample_weight=importance)
loss['Not weighted'] = MSE(y_true, y_pr, squared=False)
loss['no reviews'] = MSE(y_true[X_num_rev == 0], y_pr[X_num_rev == 0], squared=False)
loss['1 review'] = MSE(y_true[X_num_rev == 1], y_pr[X_num_rev == 1], squared=False)
loss['over 2 review'] = MSE(y_true[X_num_rev >= 2], y_pr[X_num_rev >= 2], squared=False)
loss['over 5 review'] = MSE(y_true[X_num_rev >= 5], y_pr[X_num_rev >= 5], squared=False)
loss['over 10 review'] = MSE(y_true[X_num_rev >= 10], y_pr[X_num_rev >= 10], squared=False)
loss['over 20 review'] = MSE(y_true[X_num_rev >= 20], y_pr[X_num_rev >= 20], squared=False)

print (loss)
loss.to_csv('models/sum_results.csv')

test_results = pd.DataFrame(dtype="float64");
test_results['True'] = y_true
test_results['Predicted'] = y_pr
test_results['Num Users'] = X_num_rev

test_results.to_csv('models/full_results.csv')
