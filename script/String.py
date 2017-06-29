import numpy as np
import pandas as pd


######################################
# To find whether csv contain string
######################################

# houseprice=pd.read_csv('train.csv')
houseprice=pd.read_csv('../data/RankMiss.csv')
count = houseprice['SalePrice'].count()
head_number = list(houseprice).__len__()
i=1
while i<head_number:
    if isinstance(houseprice[list(houseprice)[i]][0],str):
        print houseprice[list(houseprice)[i]].value_counts()
        list(houseprice)[i]
    i += 1
# print houseprice['Condition2'].value_counts()
# column = 'Condition2'
# i = 0
# j = 0
# size = houseprice[column].value_counts().count()
# list = houseprice[column].value_counts().index.tolist()
# print list
print houseprice
