import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


train=pd.read_csv("TechneX/ Data/train.csv")
test=pd.read_csv("TechneX/ Data/test.csv")

## Data Preprocessing

## Amount
print("Train Data Missing Values:",train.DUE_MORTGAGE.isnull().sum())
print("Test Data Missing Values:",test.DUE_MORTGAGE.isnull().sum())

## Due Mortgage

	## Missing Value Replacement with Median Value
train.DUE_MORTGAGE.fillna(train.DUE_MORTGAGE.median(),inplace=True)
test.DUE_MORTGAGE.fillna(test.DUE_MORTGAGE.median(),inplace=True)

## Value
train.VALUE.fillna(train.VALUE.mean(),inplace=True)
test.VALUE.fillna(test.VALUE.mean(),inplace=True)

## Reason
train.REASON.fillna(1.0,inplace=True)
test.REASON.fillna(1.0,inplace=True)

## OCC
train.OCC.fillna(0.0,inplace=True)
test.OCC.fillna(0.0,inplace=True)

## TJOB
train.TJOB.fillna(train.TJOB.mean(),inplace=True)
test.TJOB.fillna(train.TJOB.mean(),inplace=True)

## DCL
train.DCL.fillna(0.0,inplace=True)
test.DCL.fillna(0.0,inplace=True)

## CLT
train.CLT.fillna(train.CLT.median(),inplace=True)
test.CLT.fillna(train.CLT.median(),inplace=True)

## RATIO
train.RATIO.fillna(train.RATIO.mean(),inplace=True)
test.RATIO.fillna(train.RATIO.mean(),inplace=True)

## CL_COUNT
train.CL_COUNT.fillna(train.CL_COUNT.mean(),inplace=True)
test.CL_COUNT.fillna(train.CL_COUNT.mean(),inplace=True)

## Saving Cleaned Data
train.to_csv("Cleaned_Train.csv",index=False)
test.to_csv("Cleaned_Test.csv",index=False)

## Modelling Phase ##

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

## Spliting into Independent and Dependent Variable 
X=train.iloc[:,1:-1]
y=train['DEFAULTER']
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=101)

## Standardization
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

## Scoring Function
def Score(model,X_train,y_train,X_test,y_test,train=True):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import cross_val_score
    if train==True:
        print("Training Result \n")
        print("ROC AOC Value:  {0:0.4f} \n".format(roc_auc_score(y_train,model.predict(X_train))))
        scores=cross_val_score(estimator=model,X=X_train,y=y_train,cv=15,scoring='roc_auc',n_jobs=-1)
        print("Cross-Validation Score:",scores.mean())
        print("\nStandard Deviation:",scores.std())
    elif train==False:
        print("TestResult \n")
        print("ROC AOC Value:  {0:0.4f} \n".format(roc_auc_score(y_test,model.predict(X_test))))

## Logistic Regression Model

from sklearn.linear_model import LogisticRegression
model_LR=LogisticRegression()
model_LR.fit(X_train,y_train)

## Scoring Function with Logistic Regression Model

Score(model_LR,X_train,y_train,X_test,y_test,train=True)
Score(model_LR,X_train,y_train,X_test,y_test,train=False)

## Prediction

pred_LR=model_LR.predict_proba(sc.transform(test.iloc[:,2:]))[:,1]

## XGBoost Model

## Library Importing
import xgboost as xgb

## Spliting Training Data
train_val=train.iloc[:2860,]
test_val=train.iloc[2860:,]

## XGBoost Parameters

params={"learning_rate":0.1, 
        "n_estimators":1000, 
        "max_depth":8, 
        "min_child_weight":6, 
        "gamma":0.1, 
        "subsample":0.95,
        "colsample_bytree":0.95, 
        "reg_alpha":2, 
        "objective":'binary:logistic',
        "eval_metric": 'auc',
        "scale_pos_weight":1, 
        }

predictors=['AMOUNT', 
            'DUE_MORTGAGE', 
            'VALUE', 
            'REASON', 
            'OCC', 
            'TJOB',
            'DCL', 
            'CLT', 
            'CL_COUNT', 
            'RATIO', 
            'CONVICTED', 
            'VAR_1', 
            'VAR_2',
            'VAR_3']

outcome='DEFAULTER'


dtrain = xgb.DMatrix(data=train.loc[:,predictors], label= train.loc[:,outcome])
dtest = xgb.DMatrix(data=test.loc[:,predictors])
num_rounds = 10000

## XGBoost Crossvalidation
model_cv = xgb.cv(params, dtrain, num_rounds, nfold=10, early_stopping_rounds=20, verbose_eval=20)

## XGBoost Model Training
model_XGB = xgb.train(params, dtrain, num_boost_round = 238)

## XGBoost Prediction
pred_XGB=model_XGB.predict(dtest)

## Model Ensembling
final_pred=pred_XGB*0.99 + 0.01*pred_LR

## Submission
submission=pd.read_csv("TechneX/ Data/sample_submission.csv")
submission.LOAN_ID=test.LOAN_ID
submission.DEFAULTER=final_pred

submission.to_csv("Final_Predictions.csv",index=False)





