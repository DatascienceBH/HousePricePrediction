import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import Lasso
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def lasso(alpha):
    train_df = pd.read_csv('RankTrainSVD.csv')
    # houseprice= pd.read_csv('trainRankSVD.csv')
    # houseprice = pd.read_csv('data/RankMissTrainWithMostFre.csv')
    houseprice = pd.read_csv('data/train_dummy.csv')
    # houseprice = pd.read_csv('data/data_process/train_dummy.csv')



    # test_df = pd.read_csv('data/data_process/test_dummy.csv')
    # test_df = pd.read_csv('data/RankMissTestWithMostFre.csv')
    test_df = pd.read_csv('data/test_dummy.csv')


    i = 0
    count=test_df['Id'].size
    best_alpha = alpha
    # best_alpha = 0.1

    # while i<count:
    #     test_df.loc[i,'Id'] = test_df['Id'][i]+1461
    #     i+=1
    # test_df = pd.read_csv('data/data_process/test_dummy.csv')



    label_df = pd.DataFrame(index = train_df.index, columns=["SalePrice"])
    label_df["SalePrice"] = np.log(train_df["SalePrice"])
    houseprice = houseprice.drop('SalePrice',axis=1)



    # print houseprice.shape
    # print test_df.shape


    regr = Lasso(alpha=best_alpha, max_iter=50000)
    regr.fit(houseprice, label_df)


    # y = np.log1p(houseprice['SalePrice'])


    y_pred = regr.predict(houseprice)
    y_test = label_df
    # print("Lasso score on training set: ",best_alpha, rmse(y_test, y_pred))


    # plt.scatter(y_pred, np.expm1(y))
    # plt.plot([min(y_pred),max(y_pred)], [min(y_pred),max(y_pred)], c="red")
    # plt.show()
    # print y_pred


    y_pred_lasso = regr.predict(test_df)
    # y_pred = (y_pred_xgb + y_pred_lasso) / 2
    y_pred_lasso = np.exp(y_pred_lasso)
    pred_df = pd.DataFrame(y_pred_lasso, index=test_df["Id"], columns=["SalePrice"])
    pred_df.to_csv('output.csv', header=True, index_label='Id')



    # mask = regr.coef_ != 0
    # new_data = houseprice[1:, list(houseprice)[mask]]
    # print new_data
    # print new_data.shape
    # print regr.coef_ != 0

    j = 0
    coef = []

    while j<np.size(regr.coef_):
        if regr.coef_[j] != 0:
            coef.append(1)
        else:
            coef.append(0)
        j += 1

    k = 1
    new_data = []
    while k<np.size(coef):
        if (coef[k] != 0):
            new_data.append(list(houseprice)[k])
        k += 1
    # print new_data

    LassoIndex = pd.DataFrame(new_data)
    # print LassoIndex
    # LassoIndex.to_csv("data/LassoIndex2.csv")
    return [best_alpha,rmse(y_test, y_pred)]
# i = 0.00000001
# k = 0
# m = 0
# ba = 0;
# while i<0.001:
#     [ba,m] = lasso(i)
#     if m < k:
#         k = m
#     i +=0.00001



[ba,m]=lasso(0.000441359781404)

print (m,ba)