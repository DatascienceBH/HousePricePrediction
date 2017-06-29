import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import make_scorer,mean_squared_error
from sklearn.linear_model import Lasso


houseprice = pd.read_csv('RankAscend.csv')
y = np.log1p(houseprice.SalePrice)
x = np.log1p(houseprice['GrLivArea'])
x_plot = houseprice.loc[houseprice['SaleCondition'] == 1, 'GrLivArea']
y_plot = y[houseprice['SaleCondition'] == 1]

# plt.scatter(x_plot,y_plot)
# plt.show()


# alphas = [1e-4,5e-4,1e-3,5e-3]
# cv_lasso = [cross_val_score(Lasso(alpha = alpha , max_iter = 50000), houseprice , y).mean() for alpha in alphas]
# pd.Series(cv_lasso,index = alphas).plot()
# plt.show()
model_lasso = Lasso(alpha=5e-4, max_iter=50000).fit(houseprice, y)
# print model_lasso.coef_
coef = pd.Series(model_lasso.coef_, index = houseprice.columns).sort_values()
# print coef['GrLivArea']
imp_coef = pd.concat([coef.head(10),coef.tail(10)]);
imp_coef.plot(kind = "barh")
plt.title("Coefficients in the Model")
# plt.show()
p_pred = np.expm1(model_lasso.predict(houseprice))
plt.scatter(p_pred, np.expm1(y))
plt.show()
# plt.plot([min(p_pred),max(p_pred)], [min(p_pred),max(p_pred)], c="red")
# plt.show()
