# from sklearn.grid_search import GridSearchCV
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import preprocessing,utils
# import pandas as pd
# import numpy as np
# # Build a classification task using 3 informative features
#
#
# houseprice = pd.read_csv('data/train_dummy.csv')
#
# test_df = pd.read_csv('data/test_dummy.csv')
#
# label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
# label_df["SalePrice"] = np.log(houseprice["SalePrice"])
# # label_df["SalePrice"] = houseprice["SalePrice"]
#
# houseprice = np.log1p(houseprice)
# houseprice = houseprice.drop('SalePrice',axis=1)
#
# lab_enc = preprocessing.LabelEncoder()
# encode = lab_enc.fit_transform(label_df);
# utils.multiclass.type_of_target(label_df);
# utils.multiclass.type_of_target(label_df.astype('int'))
# utils.multiclass.type_of_target(encode);
#
# rfc = RandomForestRegressor(n_jobs=-1,max_features= 2 ,n_estimators=50, oob_score = True)
#
# param_grid = {
#     'n_jobs': [12,13,14],
#     'n_estimators': [500,600],
#     'max_features': [0.5, 0.6],
#     'max_depth': [4,5],
#     'min_samples_leaf': [0.1]
# }
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(houseprice, label_df)
# print CV_rfc.best_params_
# # {'max_features': 0.5, 'n_estimators': 500, 'n_jobs': 13, 'max_depth': 5, 'min_samples_leaf': 0.1}



############################################################################################################################################################
# from sklearn.grid_search import GridSearchCV
# from sklearn.datasets import make_classification
# from sklearn.ensemble import RandomForestRegressor
# from sklearn import preprocessing,utils
# import pandas as pd
# import numpy as np
# # Build a classification task using 3 informative features
#
#
# houseprice = pd.read_csv('trainXandL.csv')
# label = pd.read_csv('label.csv')
# test_df = pd.read_csv('testXandL.csv')
#
#
# label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
# label_df["SalePrice"] = label["SalePrice"]
#
#
# lab_enc = preprocessing.LabelEncoder()
# encode = lab_enc.fit_transform(label_df);
# utils.multiclass.type_of_target(label_df);
# utils.multiclass.type_of_target(label_df.astype('int'))
# utils.multiclass.type_of_target(encode);
#
# rfc = RandomForestRegressor(n_jobs=-1,max_features= 2 ,n_estimators=50, oob_score = True)
#
# param_grid = {
#     'n_jobs': [12,13,14],
#     'n_estimators': [500,600,700],
#     'max_features': [0.5, 0.6,0.7],
#     'max_depth': [4,5,6],
#     'min_samples_leaf': [0.1,0.5]
# }
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
# CV_rfc.fit(houseprice, label_df)
# print CV_rfc.best_params_
# # {'max_features': 0.5, 'n_estimators': 500, 'n_jobs': 13, 'max_depth': 5, 'min_samples_leaf': 0.1}
# # {'max_features': 0.5, 'n_estimators': 600, 'n_jobs': 12, 'max_depth': 4, 'min_samples_leaf': 0.1}



############################################################################################################################################################




from sklearn.grid_search import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing,utils
import pandas as pd
import numpy as np
from scipy.stats import skew
# Build a classification task using 3 informative features


houseprice = pd.read_csv('train.csv')
label = pd.read_csv('label.csv')


label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
label_df["SalePrice"] = np.log(houseprice["SalePrice"])

numeric_feats = houseprice.dtypes[houseprice.dtypes != "object"].index

skewed_feats = houseprice[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

houseprice[skewed_feats] = np.log1p(houseprice[skewed_feats])

# houseprice = np.log1p(houseprice)

houseprice = pd.get_dummies(houseprice)
houseprice = houseprice.fillna(houseprice.mean())

lab_enc = preprocessing.LabelEncoder()
encode = lab_enc.fit_transform(label_df);
utils.multiclass.type_of_target(label_df);
utils.multiclass.type_of_target(label_df.astype('int'))
utils.multiclass.type_of_target(encode);

rfc = RandomForestRegressor(n_jobs=-1,max_features= 2 ,n_estimators=50, oob_score = True)

param_grid = {
    'n_jobs': [12,13,14],
    'n_estimators': [500,600,700],
    'max_features': [0.5, 0.6,0.7],
    'max_depth': [4,5,6],
    'min_samples_leaf': [0.1,0.5]
}
CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 5)
CV_rfc.fit(houseprice, label_df)
print CV_rfc.best_params_
# {'max_features': 0.5, 'n_estimators': 500, 'n_jobs': 13, 'max_depth': 5, 'min_samples_leaf': 0.1}
# {'max_features': 0.5, 'n_estimators': 600, 'n_jobs': 12, 'max_depth': 4, 'min_samples_leaf': 0.1}