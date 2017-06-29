import numpy as np
import pandas as pd

################################################
# To translate string into number by rank them
################################################

# houseprice=pd.read_csv('train.csv')
houseprice=pd.read_csv('testMissing.csv')
count = houseprice['SaleCondition'].count()
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
    list = houseprice[column].value_counts().index.tolist()
    size = houseprice[column].value_counts().count()
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

        print "%s %s" %(list(houseprice)[i],judge)
        if judge:
            rankByFrequency(list(houseprice)[i])

        i += 1


# setValue(list(houseprice)[2],'RL',0)
setAllByFrequency()
# print houseprice[list(houseprice)[2]].value_counts()
#
houseprice.to_csv('testRank.csv',index=False)
#

