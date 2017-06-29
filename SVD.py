import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandas import DataFrame
houseprice=pd.read_csv('data/RankAscend.csv')
label_df = pd.DataFrame(index=houseprice.index, columns=['SalePrice'])
label_df['SalePrice']=np.log(houseprice['SalePrice'])
houseprice=houseprice.drop('SalePrice',axis=1)

houseprice=houseprice.drop('Id',axis=1)
u,sigma,vt=np.linalg.svd(houseprice)
# print(sigma)
# print(sigma[2])
sig3=np.mat([[sigma[0],0,0,0],[0,sigma[1],0,0],[0,0,sigma[2],0],[0,0,0,sigma[3]]])
#print (u[:,:3]*sig3*vt[:3,:])
hpnew=DataFrame(u[:,:4]*sig3*vt[:4,:])
print sigma.shape
hpnew.to_csv('data/trainRankSVD.csv')
