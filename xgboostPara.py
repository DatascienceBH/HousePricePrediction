from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing,utils
import pandas as pd
import xgboost as xgb
import numpy as np
# Build a classification task using 3 informative features


houseprice = pd.read_csv('xgTrain.csv')
label = pd.read_csv('xgy.csv')
test_df = pd.read_csv('xgTest.csv')


label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
label_df["SalePrice"] = label["SalePrice"]


lab_enc = preprocessing.LabelEncoder()
encode = lab_enc.fit_transform(label_df);
utils.multiclass.type_of_target(label_df);
utils.multiclass.type_of_target(label_df.astype('int'))
utils.multiclass.type_of_target(encode);

param_test2 = {
    'max_depth':[3,4,5,6,7,8,9],
    'min_child_weight':[5,6,7,8,9],
    'gamma':[i/10.0 for i in range(0,1)],
    'subsample':[i/10.0 for i in range(3,4,6)],
    'colsample_bytree':[i/10.0 for i in range(5,6)],
    'reg_alpha':[1e-5, 1e-2, 0.1, 1],
    'reg_lambda':[1e-5, 1e-2, 0.1, 1]
}


gsearch2 = GridSearchCV(estimator = xgb.XGBRegressor(colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=7,
                 min_child_weight=2,
                 n_estimators=7200,
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1),  param_grid = param_test2,n_jobs=4,iid=False, cv=5)
gsearch2.fit(houseprice, label_df)
print gsearch2.best_params_
