import numpy as np
import pandas as pd


##################################################
# To fill missing value by most frequent value
##################################################

houseprice = pd.read_csv('../data/RankMissTrain.csv')
count = houseprice['SaleCondition'].count()


def getMostFrequentValue(column):
    return houseprice[column].value_counts().index[0];


def getMissingSize(column):
    return houseprice[[column]][houseprice[column].isnull() == True].size


def imputeMissingValue(column):
    houseprice.loc[houseprice[column].isnull(),column] = getMostFrequentValue(column)


def getNullIndex(column):
    return houseprice[column].isnull()

# imputeMissingValue('LotFrontage')
# houseprice.loc[getNullIndex('LotFrontage'),'LotFrontage'] = getMostFrequentValue('LotFrontage')

#houseprice.to_csv('AvgPrice.csv',index=False)

# print list(houseprice)[0]


def imputeAll():
    length = list(houseprice).__len__()
    i = 1
    while i < length:
        imputeMissingValue(list(houseprice)[i])
        i += 1

imputeAll()
houseprice.to_csv('../data/RankMissTrainWithMostFre.csv',index=False)

