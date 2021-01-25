import numpy as np
import pandas as pd
import string
import os
import json
import matplotlib.pyplot as plt
import sklearn.metrics as sklmetrics
import sklearn.svm as sklsvm
from numpy import cov
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from scipy.stats import loguniform
import xgboost as xgb

pd.options.mode.chained_assignment = None  # default='warn'

df_all = pd.read_csv("titanic_train_test.csv")
features = ['Age', 'Fare', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', 'Ticket', 'Cabin', 'Name']

''' 1. FEATURE ENGINEERING '''

'''
# df_train = df_all[:891]
# df_test = df_all[891:]
# check for missing data
for col in features:
    train_na = df_train[col].isna().sum()
    if train_na > 0:
        print(f"train {col}: missing {train_na} / {df_train[col].count()}")
print()
for col in features:
    test_na = df_test[col].isna().sum()
    if test_na > 0:
        print(f"test {col}: missing {test_na} / {df_test[col].count()}")

# train Age: missing 177 / 714
# train Embarked: missing 2 / 889
# train Cabin: missing 687 / 204

# test Age: missing 86 / 332
# test Fare: missing 1 / 417
# test Cabin: missing 327 / 91
'''

# when filling in age, use median of the same Pclass b/c high correlation within group
# take Age from median of PassengerID, Age, Fare, SubSp, Parch, and Survived
age_by_pclass_sex = df_all.groupby(['Sex', 'Pclass']).median()['Age']
df_all['Age'] = df_all.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

'''
               PassengerId   Age      Fare  SibSp  Parch  Survived
Sex    Pclass
female 1               710  36.0  80.92915      0      0       1.0
       2               589  28.0  23.00000      0      0       1.0
       3               637  22.0  10.48960      0      0       0.5
male   1               648  42.0  49.50420      0      0       0.0
       2               715  29.5  13.00000      0      0       0.0
       3               649  25.0   7.89580      0      0       0.0
'''

# the 2 missing embarked have the same ticket
# by googling their name, find that "Stone, Mrs. George Nelson (Martha Evelyn)" embarked from S with Amelie Icard
df_all['Embarked'] = df_all['Embarked'].fillna('S')

# assume that Fare is related to family size (Parch and SibSp) and Pclass
median_fare = df_all.groupby(['SibSp', 'Parch', 'Pclass'])['Fare'].median()
df_all['Fare'] = df_all['Fare'].fillna(median_fare[0][0][3])

# first number of cabin number indicates which section it was in
# 100% of A-C are 1st class
# D-E have small amount of 2nd & 3rd class
# F is all 2nd & 3rd
# G has only 3rd class
# M (missing) has mostly 2nd & 3rd, some 1st
# T (1 person) is all 1st
# from A-G, increase = further away from staircase

# change empty rows to M for missing
df_all['Deck'] = df_all['Cabin'].astype(str).str[0]
df_all['Deck'].loc[df_all['Deck'] == 'n'] = 'M'
# change deck T to A because similar
df_all['Deck'].loc[df_all['Deck'] == 'T'] = 'A'
df_all = df_all.drop(['Cabin'], axis = 1)

# survival rate is different for each deck, so can't discard info
# survival for M lowest, possibly because harder to retrieve that data for victims
# group decks together by similar features
# now finished filling in missing values
df_all['Deck'] = df_all['Deck'].replace(['A', 'B', 'C'], 'ABC')
df_all['Deck'] = df_all['Deck'].replace(['D', 'E'], 'DE')
df_all['Deck'] = df_all['Deck'].replace(['F', 'G'], 'FG')

'''
print(df_all['Deck'].value_counts())
# M      1014
# ABC     182
# DE       87
# FG       26
'''

df_all['Fare'] = pd.qcut(df_all['Fare'], 13)    # split fare into 13 bins
df_all['Age'] = pd.qcut(df_all['Age'], 10)      # split age into 10 bins

# different family size (SibSp + Parch) have different survival rates
    # size = 1 --> alone
    # size = 2, 3, 4 --> small
    # size = 5, 6 --> medium
    # size >= 7 --> large
df_all['Family_Size'] = df_all['SibSp'] + df_all['Parch'] + 1
family_map = {1: 'Alone',
              2: 'Small', 3: 'Small', 4: 'Small',
              5: 'Medium', 6: 'Medium',
              7: 'Large', 8: 'Large', 11: 'Large'}
df_all['Family_Size_Grouped'] = df_all['Family_Size'].map(family_map)

# Many non-family groups used the same ticket number
# count frequency of each ticket number
df_all['Ticket_Frequency'] = df_all.groupby('Ticket')['Ticket'].transform('count')

# get title in front of name (Miss, Mr, Mrs, Col, Dr etc.)
# note that first names are sometimes split as the title
df_all['Title'] = df_all['Name'].str.split(', ', expand = True)[1].str.split('.', expand = True)[0]
df_all['Title'] = df_all['Title'].replace(['Miss', 'Mrs', 'Ms', 'Mlle', 'Lady', 'Mme', 'the Countess', 'Dona'], 'Miss/Mrs/Ms')
df_all['Title'] = df_all['Title'].replace(['Dr', 'Col', 'Major', 'Jonkheer', 'Capt', 'Sir', 'Don', 'Rev'], 'Dr/Military/Noble/Clergy')

# Mrs (married) has the highest survival rate among females
df_all['Is_Married'] = 0
df_all['Is_Married'].loc[df_all['Title'] == 'Mrs'] = 1

# return list of families based on the name column
# uncleaned format: "last, title. first"
def extract_surname(data):
    families = []
    
    for i in range(len(data)):
        name = data.iloc[i]
        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name

        family = name_no_bracket.split(',')[0]

        for c in string.punctuation:
            family = family.replace(c, '').strip()

        families.append(family)
    return families

df_all['Family'] = extract_surname(df_all['Name'])

# split the dataset
df_train = df_all[:891]
df_test = df_all[891:]

# find common family names between train and test sets
# then calculate Family_Survival_Rate for families w/ > 1 members
# also create Family_Survival_Rate_NA for non-unique families
    # implies that survival rate is not applicable to them
# also create ticket_Survival_Rate and Ticket_Survival_Rate_NA w/ same method
# Ticket_Survival_Rate and Family_Survival_Rate averaged, becomes Survival_Rate
    # also average _NA
    # reduces dimensionality
non_unique_families = [x for x in df_train['Family'].unique() if x in df_test['Family'].unique()]
non_unique_tickets = [x for x in df_train['Ticket'].unique() if x in df_test['Ticket'].unique()]

# gets median of survived and family size for each family surname
df_family_survival_rate = df_train.groupby('Family')[['Survived', 'Family', 'Family_Size']].median()
df_ticket_survival_rate = df_train.groupby('Ticket')[['Survived', 'Ticket', 'Ticket_Frequency']].median()
family_rates = {}
ticket_rates = {}

# check whether a family exists in both train and test, and has > 1 members
for i in range(len(df_family_survival_rate)):
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:  # [, 1] is Family_Size
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]     # [, 0] is the median survival rate

# check whether a ticket exists in both train and test, and has > 1 members
for i in range(len(df_ticket_survival_rate)):
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:  # [, 1] is Ticket_Frequency
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]     # [, 0] is the median survival rate

mean_survival_rate = np.mean(df_train['Survived'])

train_family_survival_rate = []
train_family_survival_rate_NA = []
test_family_survival_rate = []
test_family_survival_rate_NA = []

# whether family survival rate is not applicable (NA) to that passenger
for i in range(len(df_train)):
    # do take into account family's median survival rate
    if df_train['Family'][i] in family_rates:
        train_family_survival_rate.append(family_rates[df_train['Family'][i]])
        train_family_survival_rate_NA.append(1)
    # not enough data on that family; not applicable
    else:
        train_family_survival_rate.append(mean_survival_rate)
        train_family_survival_rate_NA.append(0)

# same thing but for test set instead of train set
for i in range(len(df_test)):
    if df_test['Family'].iloc[i] in family_rates:
        test_family_survival_rate.append(family_rates[df_test['Family'].iloc[i]])
        test_family_survival_rate_NA.append(1)
    else:
        test_family_survival_rate.append(mean_survival_rate)
        test_family_survival_rate_NA.append(0)

df_train['Family_Survival_Rate'] = train_family_survival_rate
df_train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
df_test['Family_Survival_Rate'] = test_family_survival_rate
df_test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

# Same as block above but for ticket instead of family
train_ticket_survival_rate = []
train_ticket_survival_rate_NA = []
test_ticket_survival_rate = []
test_ticket_survival_rate_NA = []

for i in range(len(df_train)):
    if df_train['Ticket'][i] in ticket_rates:
        train_ticket_survival_rate.append(ticket_rates[df_train['Ticket'][i]])
        train_ticket_survival_rate_NA.append(1)
    else:
        train_ticket_survival_rate.append(mean_survival_rate)
        train_ticket_survival_rate_NA.append(0)

for i in range(len(df_test)):
    if df_test['Ticket'].iloc[i] in ticket_rates:
        test_ticket_survival_rate.append(ticket_rates[df_test['Ticket'].iloc[i]])
        test_ticket_survival_rate_NA.append(1)
    else:
        test_ticket_survival_rate.append(mean_survival_rate)
        test_ticket_survival_rate_NA.append(0)

df_train['Ticket_Survival_Rate'] = train_ticket_survival_rate
df_train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
df_test['Ticket_Survival_Rate'] = test_ticket_survival_rate
df_test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

# create new survival rate feature by averaging family and ticket rates
for df in [df_train, df_test]:
    df['Survival_Rate'] = (df['Family_Survival_Rate'] + df['Ticket_Survival_Rate'])/2
    df['Survival_Rate_NA'] = (df['Family_Survival_Rate_NA'] + df['Ticket_Survival_Rate_NA'])/2
    
''' 2. FEATURE TRANSFORMATION '''

# LabelEncoder labels classes from 0 to n
# convert category-type variables to numberical labels
# ex. LE.fit([1, 1, 2, 6]) --> classes [1, 2, 6]
# ex. LE.transform([1, 1, 2, 6]) --> labels [0, 0, 1, 2]
non_numeric_features = ['Age', 'Fare', 'Sex', 'Family_Size_Grouped', 'Embarked', 'Deck', 'Title']
for df in [df_train, df_test]:
    for feature in non_numeric_features:
        df[feature] = LabelEncoder().fit_transform(df[feature])

# One Hot Encoding expands categorial features (Pclass, Sex, Deck, Embarked, Title)
# Age and Fare not converted, because they're ordinal
cat_features = ['Pclass', 'Sex', 'Family_Size_Grouped', 'Embarked', 'Deck', 'Title']
encoded_features = []
for df in [df_train, df_test]:
    for feature in cat_features:
        # one-hot transform one feature from list
        encoded_feat = OneHotEncoder().fit_transform(df[feature].values.reshape(-1, 1)).toarray()
        # number of classes it split into
        n = df[feature].nunique()
        # new column names feature_1, feature_2, ...
        cols = ['{}_{}'.format(feature, n) for n in range(1, n + 1)]
        # format expanded feature into a dataframe
        encoded_df = pd.DataFrame(encoded_feat, columns = cols)
        encoded_df.index = df.index
        # add expanded dataframe to list of processed features
        encoded_features.append(encoded_df)

# merge new features with train & test dataframes horizontally
# first 6 in list for df_train, last 6 for df_test
df_train = pd.concat([df_train, *encoded_features[:6]], axis = 1)
df_test = pd.concat([df_test, *encoded_features[6:]], axis = 1)

# drop useless columns
df_all = pd.concat([df_train, df_test], sort = True).reset_index(drop = True)
drop_cols = ['PassengerId', 'Sex', 'SibSp', 'Parch', 'Pclass', 'Embarked', 
             'Ticket', 'Name', 'Survived', 'Deck', 'Family_Size', 'Family_Size_Grouped', 
             'Title', 'Family', 'Family_Survival_Rate', 'Family_Survival_Rate_NA', 
             'Ticket_Survival_Rate', 'Ticket_Survival_Rate_NA']
df_all.drop(columns = drop_cols, inplace = True)
trainY = df_train['Survived'].values    # get Survived before it gets dropped
Ids = df_test['PassengerId'].values     # get PassengerIds   
df_train.drop(columns = drop_cols, inplace = True)
df_test.drop(columns = drop_cols, inplace = True)

''' 3. MODEL TRAINING '''

# normalize inputs with StandardScaler
trainX = pd.DataFrame(StandardScaler().fit_transform(df_train), index = df_train.index, columns = df_train.columns)
testX = pd.DataFrame(StandardScaler().fit_transform(df_test), index = df_test.index, columns = df_test.columns)

''' CHECK FEATURE IMPORTANCE
model_rf = RandomForestRegressor()
model_rf.fit(trainX, trainY)
ind =  model_rf.feature_importances_.argsort()
plt.barh(trainX.columns[ind], model_rf.feature_importances_[ind])
plt.show()
'''

'''
# SibSp > 4, Parch > 2 useless, so set them to 4, 2 respectively to reduce number of features
df_train.loc[df_train.SibSp >= 4, 'SibSp'] = 4
df_train.loc[df_train.Parch >= 2, 'Parch'] = 2
df_test.loc[df_test.SibSp >= 4, 'SibSp'] = 4
df_test.loc[df_test.Parch >= 2, 'Parch'] = 2
'''

def randomTreeSearch():
    paramRandom = {'n_estimators': range(500, 2000),
                   'max_depth': [4, 5, 6, 7, 8, 9],
                   'max_features': ['auto'],
                   'random_state': [1, 2, 3],
                   'min_samples_leaf': [3, 4, 5, 6, 7],
                   'min_samples_split': [2, 3, 4, 5, 6],
                   'oob_score': [True],
                   'bootstrap': [True]}
    model = RandomizedSearchCV(RandomForestClassifier(), paramRandom, cv = 5, verbose = 4, n_iter = 1500, n_jobs = -2)
    return model

def randomExtraTrees():
    paramRandom = {'n_estimators': range(100, 1500),
                   'max_depth': [3, 4, 5, 6, 7],
                   'max_features': ['auto'],
                   'random_state': [1, 2, 3],
                   'min_samples_leaf': [3, 4, 5, 6, 7],
                   'min_samples_split': [2, 3, 4, 5, 6],
                   'oob_score': [True],
                   'bootstrap': [True]}
    model = RandomizedSearchCV(ExtraTreesClassifier(), paramRandom, cv = 5, verbose = 4, n_iter = 1500, n_jobs = -2)
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
    model = RandomizedSearchCV(xgb.XGBClassifier(use_label_encoder=False), paramRandom, n_iter = 1500, cv = 5, n_jobs = -2, verbose = 4)
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

predictions = majority_vote().astype(int)
output = pd.DataFrame({'PassengerId': Ids,
                       'Survived': predictions})
output.to_csv("submission.csv", index = False)
print("saved")