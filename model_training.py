import pandas as pd
import numpy as np
import xgboost as xg
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

#load file
X = pd.read_csv('data/data_process/train_dummy.csv')
X.drop(['Id'],axis=1,inplace=True)
#print(X_train.describe())
print(X.shape)
y = pd.read_csv('data/data_process/train_house.csv')['SalePrice']

#split data and standardscale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=33)

scalerX = StandardScaler().fit(X_train)
X_train = scalerX.transform(X_train)
X_test = scalerX.transform(X_test)

scalery = StandardScaler().fit(y_train)
y_train = scalery.transform(y_train)
y_test = scalery.transform(y_test)

from sklearn.cross_validation import *

def train_and_evaluate(clf, X_train, y_train):
    cv = KFold(X_train.shape[0], 5, shuffle=True, random_state=33)
    scores = cross_val_score(clf, X_train, y_train, cv=cv)
    print ('Average coefficient of determination using 5-fold cross validation:', np.mean(scores))
    print(scores)

# SVM regressor
from sklearn.svm import SVR

clf_svr = SVR(kernel='linear')
train_and_evaluate(clf_svr, X_train, y_train)

# RBF
clf_svr_rbf = SVR(kernel='rbf')
train_and_evaluate(clf_svr_rbf, X_train, y_train)

# TreeRegression
from sklearn import ensemble
clf_et = ensemble.ExtraTreesRegressor()
train_and_evaluate(clf_et, X_train, y_train)

# try it on testdata
clf_et.fit(X_train, y_train)
print(clf_et.score(X_test, y_test))