import numpy as np
import pandas as pd


houseprice=pd.read_csv('test.csv')
count = houseprice['SaleCondition'].count()
print type(houseprice['Alley'][2])