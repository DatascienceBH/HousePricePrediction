import numpy as np
import pandas as pd
import math
################################################
# To translate string into number by rank them
################################################

# houseprice=pd.read_csv('train.csv')
houseprice=pd.read_csv('data/train.csv')
count = houseprice['Id'].count()

head_number = list(houseprice).__len__()

def cat_exploration(column):                        ## finding all categories of one feature and calculate it
    return houseprice[column].value_counts()


def setValue(column,replaced,value):
    i = 0
    size = houseprice[column].value_counts().count()
    count = houseprice[column].value_counts()[replaced]
    while i<size:
        i+=1
        houseprice.loc[houseprice[column][houseprice[column] == replaced].index,column] = value


def rankByFrequency(column):
    i = 0
    size = houseprice[column].value_counts().count()
    list = houseprice[column].value_counts().index.tolist()
    while i<size:
        houseprice.loc[houseprice[column][houseprice[column] == list[i]].index, column] = i+1
        # setValue(column,list[i], i+1)
        i += 1


def getRankValue(column,key):
    return houseprice[column].value_counts().tolist()


def setAllByFrequency():
    i = 1
    while i < head_number:
        judge = isinstance(houseprice[list(houseprice)[i]][0], str)
        mat = math.isnan(houseprice[list(houseprice)[6]][0])
        if judge:
            rankByFrequency(list(houseprice)[i])
        else:
            if mat:
                j = 1
                while j < count:
                    ju = isinstance(houseprice[list(houseprice)[i]][j], str)
                    if ju:
                        rankByFrequency(list(houseprice)[i])
                        break
                    j+=1
        i += 1



setAllByFrequency()

#
houseprice.to_csv('data/RankMissTrain.csv',index=False)
#

# print math.isnan(houseprice[list(houseprice)[6]][0])
