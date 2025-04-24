import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import ensemble


units23=pd.read_csv('/Users/phillysciastanley/Downloads/allunits_puf_23.csv')
occupied23=pd.read_csv('/Users/phillysciastanley/Downloads/occupied_puf_23.csv')
person23=pd.read_csv('/Users/phillysciastanley/Downloads/person_puf_23.csv')

units21=pd.read_csv('/Users/phillysciastanley/Downloads/allunits_puf_21.csv')
occupied21=pd.read_csv('/Users/phillysciastanley/Downloads/occupied_puf_21.csv')
person21=pd.read_csv('/Users/phillysciastanley/Downloads/person_puf_21.csv')

# print(units23.shape, occupied23.shape, person23.shape)
# print(units21.shape, occupied21.shape, person21.shape)

ocols=['HHINC_REC1','HH62PLUS','NABETHEN_RATE','NABENOW_RATE', 'HHUNDER18', 'HHUNDER6','HHSIZE','GRENT','SAFETY_RATE', 'RENT_AMOUNT','RECENT_DOWNPAY','TOTAL_FIRSTMORT','TENURE','MUTIL', 'HHFIRSTMOVEIN']
filtering=(occupied23['NABETHEN_RATE']!= -2)&(occupied23['NABETHEN_RATE']!=-1)&(occupied23['NABENOW_RATE']!=-1)
oc23=occupied23.loc[filtering,ocols+['CONTROL']].copy()
# print(oc23.columns)

pcols=['GENDER_P','EDATTAIN_P','INC_EARNINGS_P','WORKJOBS_P','TOTAL_INC_P']
filtering=(person23['TOTAL_INC_P']!=2222222222)&(person23['EDATTAIN_P']!=-2)&(person23['WORKJOBS_P']!=-2)&(person23['INC_EARNINGS_P']!=-2)
pc23=person23.loc[filtering,pcols+['CONTROL']].copy()



#gentried if neighbirhood rating now is higher than then - 0
#not gentried if now rating is lower than - 1
oc23['genIndicator']=0
for i,r in oc23.iterrows():
    if r['NABENOW_RATE']<r['NABETHEN_RATE']:
        oc23.loc[i,'genIndicator']=0
    else:
        oc23.loc[i,'genIndicator']=1


#prescence of young and old- 0 if no, 1 yes
oc23['youngIndicator']=0
oc23['oldIndicator']=0
for i,r in oc23.iterrows():
    if r['HHUNDER18']==1 or r['HHUNDER6']==1:
        oc23.loc[i,'youngIndicator']=1
    if r['HH62PLUS']==1:
        oc23.loc[i,'oldIndicator']=1

#gender- 0 if no, 1 yes
pc23['isMale']=0
pc23['isFemale']=0
pc23['otherGen']=0
for i,r in pc23.iterrows():
    if r['GENDER_P']=='Male':
        pc23.loc[i,'isMale']=1
    elif r['GENDER_P']=='Female':
        pc23.loc[i,'isFemale']=1
    else:
        pc23.loc[i,'otherGen']=1

oc23=oc23.drop(columns=['HH62PLUS', 'HHUNDER18', 'HHUNDER6'])
pc23=pc23.drop(columns=['GENDER_P'])


#random forest
# np.random.shuffle(DATA)
# X = DATA[:,:-1]
# y = DATA[:,-1]



#Splitting into train, validation, test
# Split data into training data and testing data
# Split data into training+validation and test sets (85% train+validation, 15% test)
# X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, random_state=0)
# # Split training+validation set into training and validation sets (70% train, 15% validation)
# X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.18, random_state=0) # 0.15 / 0.85 = 0.18 OR 0.18 x 0.85 = 0.15
# # Scaling the data
# scaler = StandardScaler()
# scaler.fit(X_train, y_train)
# X_train = scaler.transform(X_train)
# X_val = scaler.transform(X_val)
# X_test = scaler.transform(X_test)


# forestModel = ensemble.RandomForestClassifier(random_state=0) 

 


    
