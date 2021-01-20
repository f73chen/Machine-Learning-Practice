import numpy as np
import pandas as pd
import os
import sklearn.metrics as sklmetrics
import sklearn.svm as sklsvm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform

train_data = pd.read_csv("train_and_test2.csv")[:891]
test_data = pd.read_csv("train_and_test2.csv")[891:]

features = ['Age', 'Fare', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked0', 'Embarked1', 'Embarked2']

trainX = pd.get_dummies(train_data[features])
trainY = train_data['Survived']
testX = pd.get_dummies(test_data[features])

paramRandom = {'n_estimators': range(5, 200),
               'max_depth': range(3, 10),
               'criterion': ['gini', 'entropy']
               'random_state': [1],
               'max_features': ['auto', 'log2']}

model = RandomizedSearchCV(RandomForestClassifier(), paramRandom, cv = 5, refit = True, verbose = 4, n_iter = 300, n_jobs = -2)

model.fit(trainX, trainY)
predictions = model.predict(testX)

output = pd.DataFrame({'PassengerId': test_data.PassengerId,
                       'Survived': predictions})
output.to_csv("submission.csv", index = False)
print("saved")