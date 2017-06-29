import pandas as pd
import numpy as np

# import data and delete the column of ID
dataset_path = './total.csv'
data=pd.read_csv(dataset_path)
# data.drop(['Id'],axis=1, inplace=True)
feature_names = data.columns[0:].tolist()
print('data_shape', data.shape)
print('there are {} items'.format(data.shape[0]))
print('there are {} charicteristics:{}'.format(len(feature_names),feature_names))

# features which contain missing values and its number
missing_No = {}
for feature in feature_names:
    if len(data[feature][data[feature].isnull()])>0:
        print(feature)
        print(len(data[feature][data[feature].isnull()]))
        missing_No[feature] = len(data[feature][data[feature].isnull()])
print(missing_No)

# filling the missing value with average value in the numerical feature
for key in missing_No:
    if data[key].dtype !='object':
        print(key)
        data[key][data[key].isnull()]=data[key].mean()

#filling the missing value with 'U0' in category feature
for key in missing_No:
    if data[key].dtype =='object':
        data[key][data[key].isnull()]='U0'

#replace the text with number in descending for ranking feature
comap1={
    'Ex':1,
    'Gd':2,
    'TA':3,
    'Fa':4,
    'Po':5,
    'U0':6}
for feature in ['ExterQual','ExterCond','HeatingQC','KitchenQual',
                'BsmtQual','BsmtCond','FireplaceQu','GarageQual',
                'GarageCond','PoolQC']:
    data[feature]=data[feature].map(comap1)

comap2={
    'Gd':1,
    'Av':2,
    'Mn':3,
    'No':4,
    'U0':5}
data['BsmtExposure']=data.BsmtExposure.map(comap2)

comap3={'GLQ':1,'ALQ':2, 'BLQ':3, 'Rec':4, 'LwQ':5,'Unf':6, 'U0':7}
data['BsmtFinType1']=data.BsmtFinType1.map(comap3)
data['BsmtFinType2']=data.BsmtFinType2.map(comap3)
comap4={'Fin':1,'RFn':2, "Unf":3, 'U0':4}
data['GarageFinish']=data.GarageFinish.map(comap4)
comap5={'GdPrv':1, 'MnPrv':2, 'GdWo':3,'MnWw':4,'U0':5}
data['Fence']=data.Fence.map(comap5)

#get dummies and output to csv
data_dummy = pd.get_dummies(data)

train_dummy =data_dummy.loc[0:1459]
test_dummy = data_dummy.loc[1460:]

train_dummy.to_csv('./train_dummy.csv',index=False,header=True)
test_dummy.to_csv('./test_dummy.csv',index=False,header=True)