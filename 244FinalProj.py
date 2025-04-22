import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



units23= pd.read_csv('/Users/phillysciastanley/Downloads/allunits_puf_23.csv')
occupied23= pd.read_csv('/Users/phillysciastanley/Downloads/occupied_puf_23.csv')
person23= pd.read_csv('/Users/phillysciastanley/Downloads/person_puf_23.csv')

units21= pd.read_csv('/Users/phillysciastanley/Downloads/allunits_puf_21.csv')
occupied21= pd.read_csv('/Users/phillysciastanley/Downloads/occupied_puf_21.csv')
person21= pd.read_csv('/Users/phillysciastanley/Downloads/person_puf_21.csv')

# print(units23.shape, occupied23.shape, person23.shape)
# print(units21.shape, occupied21.shape, person21.shape)


ocols= ['HHINC_REC1','HH62PLUS','NABETHEN_RATE','NABENOW_RATE', 'HHUNDER18', 'HHUNDER6','HHSIZE','GRENT','SAFETY_RATE', 'RENT_AMOUNT','RECENT_DOWNPAY','TOTAL_FIRSTMORT','TENURE','MUTIL', 'HHFIRSTMOVEIN']
oc23= occupied23[occupied23['NABETHEN_RATE']!= -2][occupied23['NABETHEN_RATE']!=-1][occupied23['NABENOW_RATE']!=-1][ocols+['CONTROL']].copy()
# print(oc23.columns)

#gentried if neighbirhood rating now is higher than then - 0
#not gentried if now rating is lower than - 1
for i,r in oc23.iterrows():
    if r['NABENOW_RATE']<r['NABETHEN_RATE']:
        oc23['genIndicator']=0
    if r['NABENOW_RATE']>=r['NABETHEN_RATE']:
        oc23['genIndicator']=1