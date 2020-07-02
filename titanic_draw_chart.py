#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# 변수 탐색 코드
#-------------------------------------------------------

#############################################
#### 1. Process
#############################################

"""
A. 데이터분석

 Descriptive 데이터분석 : 독립변수 탐색 및 분석
 B. ML 모델 만들기

 Model Training 9종
 모델 평가
 주성분 분석
"""
 
#############################################
#### 2. What data do we have?
#############################################
# Import the basic python libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style='white', context='notebook', palette='deep')
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir('/home/cdsw/titanic_example')

# Read the datasets
train = pd.read_csv("./data/titanic_train.csv")
test = pd.read_csv("./data/titanic_test.csv")
IDtest = test["PassengerId"]
train.info()
test.info()


#############################################
#### 3. Exploratory data analysis
#############################################
# Check missing values in train data set
train_na = (train.isnull().sum() / len(train)) * 100
train_na = train_na.drop(train_na[train_na == 0].index).sort_values(ascending=False)[:30]
miss_train = pd.DataFrame({'Train Missing Ratio' :train_na})
miss_train.head()

# Check missing values in train data set
test_na = (test.isnull().sum() / len(test)) * 100
test_na = test_na.drop(test_na[test_na == 0].index).sort_values(ascending=False)[:30]
miss_test = pd.DataFrame({'Test Missing Ratio' :test_na})
miss_test.head()

# Fill empty and NaNs values with NaN
train = train.fillna(np.nan)
test = test.fillna(np.nan)

# Analyze the count of survivors by Pclass
ax = sns.countplot(x="Pclass", hue="Survived", data=train)
train[['Pclass', 'Survived']].groupby(['Pclass']).count().sort_values(by='Survived', ascending=False)

# Analyze the Survival Probability by Pclass
g = sns.barplot(x="Pclass",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[['Pclass', 'Survived']].groupby(['Pclass']).mean().sort_values(by='Survived', ascending=False)


# Count the number of passengers by gender
ax = sns.countplot(x="Sex", hue="Survived", data=train)
# Analyze survival count by gender
train[["Sex", "Survived"]].groupby(['Sex']).count().sort_values(by='Survived', ascending=False)


# Analyze the Survival Probability by Gender
g = sns.barplot(x="Sex",y="Survived",data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex", "Survived"]].groupby(['Sex']).mean().sort_values(by='Survived', ascending=False)



# Let's explore the distribution of age by response variable (Survived)
fig = plt.figure(figsize=(10,8),);axis = sns.kdeplot(train.loc[(train['Survived'] == 1),'Age'].dropna(), color='g',shade=True, label='Survived');axis = sns.kdeplot(train.loc[(train['Survived'] == 0),'Age'].dropna(), color='b',shade=True,label='Did Not Survived');plt.title('Age Distribution - Surviver V.S. Non Survivors', fontsize = 20);plt.xlabel("Passenger Age", fontsize = 12);plt.ylabel('Frequency', fontsize = 12);

sns.lmplot('Age','Survived',data=train)

# We can also say that the older the passenger the lesser the chance of survival


# Analyze the count of survivors by SibSP

ax = sns.countplot(x="SibSp", hue="Survived", data=train)
train[['SibSp', 'Survived']].groupby(['SibSp']).count().sort_values(by='Survived', ascending=False)


# Analyze probability of survival by SibSP
g  = sns.factorplot(x="SibSp",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["SibSp", "Survived"]].groupby(['SibSp']).mean().sort_values(by='Survived', ascending=False)


# Analyze the count of survivors by Parch

ax = sns.countplot(x="Parch", hue="Survived", data=train)
train[['Parch', 'Survived']].groupby(['Parch']).count().sort_values(by='Survived', ascending=False)

# Analyze the Survival Probability by Parch
g  = sns.factorplot(x="Parch",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("Survival Probability")
train[["Parch", "Survived"]].groupby(['Parch']).mean().sort_values(by='Survived', ascending=False)


from scipy import stats
from scipy.stats import norm, skew #for some statistics
(mu, sigma) = norm.fit(train['Fare'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
sns.distplot(train['Fare'] , fit=norm);plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best');plt.ylabel('Frequency');plt.title('Fare distribution')

# Let's check the unique values
train['Cabin'].unique()

# Analyze the count of survivors by Embarked variable
ax = sns.countplot(x="Embarked", hue="Survived", data=train)
train[['Embarked', 'Survived']].groupby(['Embarked']).count().sort_values(by='Survived', ascending=False)

# Analyze the Survival Probability by Embarked
g  = sns.factorplot(x="Embarked",y="Survived",data=train,kind="bar", size = 7 ,palette = "muted")
g.despine(left=True)
g = g.set_ylabels("survival probability")
train[["Embarked", "Survived"]].groupby(['Embarked']).mean().sort_values(by='Survived', ascending=False)



# Age, Pclass & Survival
sns.lmplot('Age','Survived',data=train,hue='Pclass')

# Age, Embarked, Sex, Pclass
g = sns.catplot(x="Age", y="Embarked",  hue="Sex", row="Pclass",   data=train[train.Embarked.notnull()], 
orient="h", height=2, aspect=3, palette="Set3",  kind="violin", dodge=True, cut=0, bw=.2)

# Relation among Pclass, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Pclass", data=train, saturation=.5, kind="bar", ci=None, aspect=.6)

# Relation among SibSP, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="SibSp", data=train, saturation=.5,kind="bar", ci=None, aspect=.6)

# Relation among Parch, Gender & Survival Rate
g = sns.catplot(x="Sex", y="Survived", col="Parch", data=train, saturation=.5,kind="bar", ci=None, aspect=.6)



#############################################
#### 4. Data preparation including feature engineering
#############################################

# Let's combining train & test for quick feature engineering. 
# Variable source is a kind of tag which indicates data source in combined data
train['source']='train'
test['source']='test'
combdata = pd.concat([train, test],ignore_index=True)
print (train.shape, test.shape, combdata.shape)

# PassengerID - Drop PassengerID
combdata.drop(labels = ["PassengerId"], axis = 1, inplace = True)

# Pclass 
combdata['Pclass'].unique()

# Name - Extract Salutation from Name variable
salutation = [i.split(",")[1].split(".")[0].strip() for i in combdata["Name"]]
combdata["Title"] = pd.Series(salutation)
combdata["Title"].unique()

# Convert other salutations to fixed Title 
combdata["Title"] = combdata["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combdata["Title"] = combdata["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
combdata["Title"] = combdata["Title"].astype(int)
combdata["Title"].unique()

combdata = pd.get_dummies(combdata, columns = ["Title"])

# Drop Name variable
combdata.drop(labels = ["Name"], axis = 1, inplace = True)


# Age

## Fill Age with the median age of similar rows according to Sex, Pclass, Parch and SibSp
# Index of NaN age rows
missing_index = list(combdata["Age"][combdata["Age"].isnull()].index)

for i in missing_index :
    median_age = combdata["Age"].median()
    filled_age = combdata["Age"][((combdata['Sex'] == combdata.iloc[i]["Sex"]) & 
                                (combdata['SibSp'] == combdata.iloc[i]["SibSp"]) & 
                                (combdata['Parch'] == combdata.iloc[i]["Parch"]) & 
                                (combdata['Pclass'] == combdata.iloc[i]["Pclass"]))].median()
    if not np.isnan(filled_age) :
        combdata['Age'].iloc[i] = filled_age
    else :
        combdata['Age'].iloc[i] = median_age
        
        
        
# Sex - Create dummy variables
#combdata["Sex"] = combdata["Sex"].map({"male": 0, "female":1}) or
combdata = pd.get_dummies(combdata, columns = ["Sex"])  

# Create a variable representing family size from SibSp and Parch
combdata["Fsize"] = combdata["SibSp"] + combdata["Parch"] + 1

# Create new feature of family size
combdata['Single'] = combdata['Fsize'].map(lambda s: 1 if s == 1 else 0)
combdata['SmallF'] = combdata['Fsize'].map(lambda s: 1 if  s == 2  else 0)
combdata['MedF'] = combdata['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
combdata['LargeF'] = combdata['Fsize'].map(lambda s: 1 if s >= 5 else 0)

# Drop Name variable
combdata.drop(labels = ["Fsize"], axis = 1, inplace = True)

# SibSp - Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["SibSp"])

# Parch - Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Parch"])

# Ticket - Extracting the ticket prefix. This might be a representation of class/compartment.
# If there is no prefix replace with U (Unknown). 

Ticket = []
for i in list(combdata.Ticket):
    if not i.isdigit() :
        Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
    else:
        Ticket.append("U")
        
combdata["Ticket"] = Ticket
combdata["Ticket"].unique()

combdata = pd.get_dummies(combdata, columns = ["Ticket"], prefix="T")

# Fare - Check the number of missing value
combdata["Fare"].isnull().sum()

# Only 1 value is missing so we will fill the same with median
combdata["Fare"] = combdata["Fare"].fillna(combdata["Fare"].median())

# Use the numpy fuction log1p which  applies log(1+x) to all elements of the column
combdata["Fare"] = np.log1p(combdata["Fare"])


# Cabin - Replace the missing Cabin number by the type of cabin unknown 'U'
combdata["Cabin"] = pd.Series([i[0] if not pd.isnull(i) else 'U' for i in combdata['Cabin'] ])


# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Cabin"], prefix="Cabin")

# Embarked - Find the number of missing values
combdata["Embarked"].isnull().sum()

# Fill Embarked missing values of dataset set with mode 'S'
combdata["Embarked"] = combdata["Embarked"].fillna("S")

# Create dummy variables
combdata = pd.get_dummies(combdata, columns = ["Embarked"], prefix="Emb")

# Create dummies for PClass Now
combdata = pd.get_dummies(combdata, columns = ["Pclass"], prefix="Pclass")

combdata.info()

#############################################
#### 5. Simple Model Evaluation
#############################################

# Import the required libraries
from sklearn.svm import SVC
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier

## Separate train dataset and test dataset using the index variable 'source'

train = combdata.loc[combdata['source']=="train"]
test = combdata.loc[combdata['source']=="test"]
test.drop(labels=["Survived"],axis = 1,inplace=True)
train.drop(labels=["source"],axis = 1,inplace=True)
test.drop(labels=["source"],axis = 1,inplace=True)


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels = ["Survived"],axis = 1)


# Cross validate model with Kfold stratified cross val
kfold = StratifiedKFold(n_splits=10)


# Modeling differents algorithms. 

random_state = 2

cls_dict = {"SVC":SVC(random_state=random_state),
"AdaBoost":AdaBoostClassifier(DecisionTreeClassifier(random_state=random_state),random_state=random_state,learning_rate=0.1),
"ExtraTrees":ExtraTreesClassifier(random_state=random_state),
"KNeighboors":KNeighborsClassifier(),
"DecisionTree":DecisionTreeClassifier(random_state=random_state),
"RandomForest":RandomForestClassifier(random_state=random_state),
"GradientBoosting":GradientBoostingClassifier(random_state=random_state),
"LogisticRegression":LogisticRegression(random_state = random_state),
"MultipleLayerPerceptron":MLPClassifier(random_state=random_state),
"LinearDiscriminantAnalysis":LinearDiscriminantAnalysis()}


cv_results = {}
for classifier in cls_dict.keys() :
    cv_results[classifier] = cross_val_score(cls_dict[classifier], X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4)

cv_resultv = []
for classifier in cv_results.keys():
    cv_resultv.append([classifier, cv_results[classifier].mean(), cv_results[classifier].std()])

cv_res = pd.DataFrame(cv_resultv, columns=['Algorithm', 'CrossValMeans', 'CrossValStd'])

#############################################
#### 6.  Creating a Model
#############################################
# 1. Adaboost
DTC = DecisionTreeClassifier()

adaDTC = AdaBoostClassifier(DTC, random_state=7)

ada_param_grid = {"base_estimator__criterion" : ["gini", "entropy"],
                  "base_estimator__splitter" :   ["best", "random"],
                  "algorithm" : ["SAMME","SAMME.R"],
                  "n_estimators" :[1,2],
                  "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3,1.5]}

gsadaDTC = GridSearchCV(adaDTC,param_grid = ada_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsadaDTC.fit(X_train,Y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_


# 2. ExtraTrees 
ExtC = ExtraTreesClassifier()

## Search grid for optimal parameters
ex_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}

gsExtC = GridSearchCV(ExtC,param_grid = ex_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsExtC.fit(X_train,Y_train)
ExtC_best = gsExtC.best_estimator_

# Best score
gsExtC.best_score_

# 3. RFC Parameters tunning 
RFC = RandomForestClassifier()

## Search grid for optimal parameters
rf_param_grid = {"max_depth": [None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
gsRFC = GridSearchCV(RFC,param_grid = rf_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsRFC.fit(X_train,Y_train)
RFC_best = gsRFC.best_estimator_

# Best score
gsRFC.best_score_


# 4. Gradient boosting 
GBC = GradientBoostingClassifier()
gb_param_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
              }
gsGBC = GridSearchCV(GBC,param_grid = gb_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsGBC.fit(X_train,Y_train)
GBC_best = gsGBC.best_estimator_

# Best score
gsGBC.best_score_

### 5. SVC classifier
SVMC = SVC(probability=True)
svc_param_grid = {'kernel': ['rbf'], 
                  'gamma': [ 0.001, 0.01, 0.1, 1],
                  'C': [1, 10, 50, 100,200,300, 1000]}
gsSVMC = GridSearchCV(SVMC,param_grid = svc_param_grid, cv=kfold, scoring="accuracy", n_jobs= 4, verbose = 1)
gsSVMC.fit(X_train,Y_train)
SVMC_best = gsSVMC.best_estimator_

# Best score
gsSVMC.best_score_


# Use voting classifier to combine the prediction power of all models
model_dict = {'rfc': RFC_best, 'extc': ExtC_best,
'svc': SVMC_best, 'adac': ada_best, 'gbc':GBC_best}

votingC = VotingClassifier(estimators=[i for i in model_dict.items()], voting='soft', n_jobs=4)
votingC = votingC.fit(X_train, Y_train)


#############################################
#### 7. Candidate Model Evaluation
#############################################
cls_dict = {"SVC":SVMC_best,
"AdaBoost":ada_best,
"ExtraTrees":ExtC_best,
"RandomForest":RFC_best,
"GradientBoosting":GBC_best}

cv_results = {}
for classifier in cls_dict.keys() :
    cv_results[classifier] = cross_val_score(cls_dict[classifier], X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4)

cv_resultv = []
for classifier in cv_results.keys():
    cv_resultv.append([classifier, cv_results[classifier].mean(), cv_results[classifier].std()])

cv_res = pd.DataFrame(cv_resultv, columns=['Algorithm', 'CrossValMeans', 'CrossValStd'])

best_single_model = cv_res.loc[cv_res['CrossValMeans'].idxmax(),'Algorithm']


cls_dict['EnsembleModel'] = votingC
cv_results['EnsembleModel'] = cross_val_score(votingC, X_train, y = Y_train, scoring = "accuracy", cv = kfold, n_jobs=4)

cv_resultv = []
for classifier in cv_results.keys():
    cv_resultv.append([classifier, cv_results[classifier].mean(), cv_results[classifier].std()])

cv_res = pd.DataFrame(cv_resultv, columns=['Algorithm', 'CrossValMeans', 'CrossValStd'])

best_model = cv_res.loc[cv_res['CrossValMeans'].idxmax(),'Algorithm']

# Save model score
cv_res.to_csv('./outpred/model_score.csv', index=False)


#############################################
#### 8. Save Model & Predict
#############################################
# Output
import pickle
filename = './model/Ensemble_Model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(votingC, f)

filename = "./model/[Best_Single_Model]_{}.pkl".format(best_single_model)
with open(filename, 'wb') as f:
    pickle.dump(cls_dict[best_single_model], f)

filename = "./model/[Best_Model]_{}.pkl".format(best_model)
with open(filename, 'wb') as f:
    pickle.dump(cls_dict[best_model], f)



# Predict and export the results
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("Ensemble_Model_Predict.csv",index=False)

# Predict and export the results
test_Survived = pd.Series(cls_dict[best_single_model].predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("[Best_Single]_{}_Predict.csv".format(best_single_model),index=False)

# Predict and export the results
test_Survived = pd.Series(cls_dict[best_model].predict(test), name="Survived")
results = pd.concat([IDtest,test_Survived],axis=1)
results.to_csv("[Best]_{}_Predict.csv".format(best_model),index=False)

results.info()