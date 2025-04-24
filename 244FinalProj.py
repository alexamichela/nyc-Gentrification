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

## Random Forest Classifier
# forestModel = ensemble.RandomForestClassifier(random_state=0) 

## Logistic Regression
# Generate features corresponding to different degree polynomial combinations and print the accuracy for each degree
#from sklearn.preprocessing import PolynomialFeatures
#from sklearn.linear_model import LogisticRegression
#for degree in range(1, 11):
#    poly = PolynomialFeatures(degree=degree)
#    # Fit and transform training data
#    X_train_poly = poly.fit_transform(X_train)

#    # Transform validation data
#    X_val_poly = poly.transform(X_val)
    
#    logreg = LogisticRegression(random_state=0, max_iter=10000)
#    logreg.fit(X_train_poly, y_train)
#    print(f"Accuracy of classifier with {degree} degree polynomial combination of features: {logreg.score(X_val_poly, y_val)}")
#    y_pred = logreg.predict(X_val_poly)
#    print(f"F1 Score on validation data with degree={degree}: {f1_score(y_val, y_pred}")

##DETERMINE WHAT DEGREE HAS THE HIGHEST ACCURACY ⇒ use that degree
#poly = PolynomialFeatures(degree=__)
## Fit and transform training data
#X_train_poly = poly.fit_transform(X_train)
## Transform validation data
#X_val_poly = poly.transform(X_val)

## Using 5-fold cross validation, tune the regularization hyperparameter C for logistic regression
#reg_strengths = [1, 3, 10, 30, 100, 300, 1000]
#for c in reg_strengths:
#    classifier = LogisticRegression(random_state=0, C=c)
#    scores = cross_val_score(classifier, X_train_poly, y_train, cv=5)
#    print(f"Average accuracy of logistic regression with C={c}: {np.mean(scores)}")

##Support Vector Machines
#from sklearn.svm import SVC
#from sklearn.metrics import f1_score
#model = SVC(random_state=0)
#model.fit(X_train, y_train)
#print(f"RBF Kernel Accuracy on validation data: {model.score(X_val, y_val)}")
#y_pred = model.predict(X_val)
#print(f"RBF Kernel F1 Score on validation data: {f1_score(y_val, y_pred}")

## Create a support vector classifier using a *linear* kernel.
## Train it on the training data and test it on the validation data.
#model2 = SVC(kernel="linear", random_state=0)
#model2.fit(X_train, y_train)
#print(f"Linear Kernel Accuracy on validation data: {model2.score(X_val, y_val)}")
#y_pred2 = model2.predict(X_val)
#print(f"Linear Kernel F1 Score on validation data: {f1_score(y_val, y_pred2}")

##MUST SEE WHICH KERNEL PRODUCES HIGHER SCORES AND THEN PUT THAT HERE
## Create a support vector classifier, train it on the training data, 
## and test it on the validation data.
## For parameter C, explore values of 1.0, 10.0, 100.0, and 1000.0
## For parameter gamma, explore values of 1.0, 10.0, 100.0, and 1000.0 
#c_values = [1.0, 10.0, 100.0, 1000.0]
#gamma_values = [1.0, 10.0, 100.0, 1000.0]
#for c in c_values:
#    for gamma in gamma_values:
#        model = SVC(C=c, gamma=gamma, kernel=__, random_state=0)
#        model.fit(X_train, y_train)
#        print(f"Accuracy on validation data with C={c} and gamma={gamma}: {model.score(X_val, y_val)}")
#        y_pred = model.predict(X_val)
#        print(f"F1 Score on validation data: {f1_score(y_val, y_pred}")
