import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import sklearn.metrics as sklmetrics
import sklearn.svm as sklsvm
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
import xgboost as xgb

train_data = pd.read_csv("train_and_test2.csv")[:891]
test_data = pd.read_csv("train_and_test2.csv")[891:]

# SibSp > 4, Parch > 2 useless, so set them to 4, 2 respectively to reduce number of features
train_data.loc[train_data.SibSp >= 4, 'SibSp'] = 4
train_data.loc[train_data.Parch >= 2, 'Parch'] = 2
test_data.loc[test_data.SibSp >= 4, 'SibSp'] = 4
test_data.loc[test_data.Parch >= 2, 'Parch'] = 2

features = ['Age', 'Fare', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked']

trainX = pd.get_dummies(train_data[features], columns = ['Pclass', 'SibSp', 'Parch', 'Embarked'])
trainY = train_data['Survived']
testX = pd.get_dummies(test_data[features], columns = ['Pclass', 'SibSp', 'Parch', 'Embarked'])

''' CHECK FEATURE IMPORTANCE
model_rf = RandomForestRegressor()
model_rf.fit(trainX, trainY)
ind =  model_rf.feature_importances_.argsort()
plt.barh(trainX.columns[ind], model_rf.feature_importances_[ind])
plt.show()
'''

def randomTreeSearch():
    paramRandom = {'n_estimators': range(200, 1000),
                   'max_depth': [5, 6, 7, 8, 9],
                   'max_features': [7, 8, 9, 10, 11, 12],
                   'random_state': [1, 2, 3],
                   "min_samples_leaf": [2, 3, 4, 5, 6],
                   "min_samples_split": [2, 3, 4, 5, 6]}
    model = RandomizedSearchCV(RandomForestClassifier(), paramRandom, cv = 5, verbose = 4, n_iter = 2000, n_jobs = -2)
    return model

def randomExtraTrees():
    paramRandom = {'n_estimators': range(100, 800),
                   'max_depth': [3, 4, 5, 6, 7],
                   'max_features': [1, 2, 3, 4, 6],
                   'random_state': [1, 2, 3],
                   "min_samples_leaf": [3, 4, 5, 6, 7],
                   "min_samples_split": [2, 3, 4, 5, 6]}
    model = RandomizedSearchCV(ExtraTreesClassifier(), paramRandom, cv = 5, verbose = 4, n_iter = 2000, n_jobs = -2)
    return model

def randomXgboost():
    paramRandom = {'booster': ['gbtree', 'gblinear'],
                   'objective': ['binary:logistic'],
                   'subsample': [0.4, 0.5, 0.6, 0.7, 0.8],
                   'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
                   'eta': [0.01, 0.03, 0.05, 0.07, 0.09],
                   'max_depth': [5, 6, 7, 8, 9],
                   'seed': [1, 2, 3, 4],
                   'eval_metric': ['logloss']}
    model = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False), paramRandom, cv = 5, n_jobs = -2, verbose = 4, n_iter = 1500)
    return model

def majority_vote():
    model_rf = randomTreeSearch()
    model_rf.fit(trainX, trainY)
    best_rf = model_rf.best_params_
    pred_rf = model_rf.predict(testX)
    
    model_et = randomExtraTrees()
    model_et.fit(trainX, trainY)
    best_et = model_et.best_params_
    pred_et = model_et.predict(testX)
    
    model_xgb = randomXgboost()
    model_xgb.fit(trainX, trainY)
    best_xgb = model_xgb.best_params_
    pred_xgb = model_xgb.predict(testX)

    scores = pd.DataFrame({'pred_rf': pred_rf,
                           'pred_et': pred_et,
                           'pred_xgb': pred_xgb})
    scores['sum'] = scores.sum(axis = 1)
    scores.loc[scores['sum'] >= 1.5, 'pred'] = 1
    scores.loc[scores['sum'] < 1.5, 'pred'] = 0

    with open("best.txt", 'a') as bestFile:
        bestFile.write("best random forest: \n")
        for k, v in best_rf.items():
            bestFile.write(str(k) + ": " + str(v) + "\n")
            
        bestFile.write("\nbest extra trees: \n")
        for k, v in best_et.items():
            bestFile.write(str(k) + ": " + str(v) + "\n")
            
        bestFile.write("\nbest xgboost: \n")
        for k, v in best_xgb.items():
            bestFile.write(str(k) + ": " + str(v) + "\n")
        bestFile.write("\n\n")

    return scores.pred

predictions = majority_vote()
output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.to_csv("submission.csv", index = False)
print("saved")