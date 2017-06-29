import pandas as pd
houseprice = pd.read_csv('RankTrainSVD.csv')

count = houseprice['SalePrice'].count()-1
head_number = list(houseprice).__len__()
i=1
while i < head_number:
    while count >= 0:
        if pd.isnull(houseprice[list(houseprice)[i]][count]):
            print list(houseprice)[i]
        count -= 1
    i += 1
    count = houseprice['SalePrice'].count()-1

# print houseprice[list(houseprice)[1]][0] is None;