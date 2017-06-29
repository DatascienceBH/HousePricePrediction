import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import Lasso, LassoCV,RidgeCV


# houseprice = pd.read_csv('data/train_dummy.csv')
#
# test_df = pd.read_csv('data/test_dummy.csv')
#
# count = test_df['Id'].size
#
# label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
# label_df["SalePrice"] = np.log(houseprice["SalePrice"])
#
# houseprice = np.log1p(houseprice)
# houseprice = houseprice.drop('SalePrice',axis=1)
#
# print label_df
# ridge = RidgeCV()
# print ridge.fit(houseprice, label_df)
# print ridge.alpha_

#################################
# houseprice = pd.read_csv('trainXandL.csv')
# label = pd.read_csv('label.csv')
# test_df = pd.read_csv('testXandL.csv')
#
#
# label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
# label_df["SalePrice"] = label["SalePrice"]
#
#
# print label_df
# ridge = RidgeCV()
# print ridge.fit(houseprice, label_df)
# print ridge.alpha_

#################################

houseprice = pd.read_csv('train.csv')


label_df = pd.DataFrame(index = houseprice.index, columns=["SalePrice"])
# label_df["SalePrice"] = houseprice["SalePrice"]
label_df["SalePrice"] = np.log(houseprice["SalePrice"])

numeric_feats = houseprice.dtypes[houseprice.dtypes != "object"].index

skewed_feats = houseprice[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index

houseprice[skewed_feats] = np.log1p(houseprice[skewed_feats])

# houseprice = np.log1p(houseprice)

houseprice = pd.get_dummies(houseprice)

#filling NA's with the mean of the column:
houseprice = houseprice.fillna(houseprice.mean())
print label_df
ridge = RidgeCV()
print ridge.fit(houseprice, label_df)
print ridge.alpha_