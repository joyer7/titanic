#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# Model Training
#-------------------------------------------------------

#############################################
# 5. Simple Model Evaluation
#############################################
# Import the required libraries
import pandas as pd
from sklearn.svm import SVC
from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
import os

## set dir
#os.chdir('/home/cdsw/titanic_example')

## Load preprocessed data & Separate train features and label 
train = pd.read_csv("./prep_data/prep_train.csv")

## Separate train features and label 
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
#### Candidate model selectd by EDA(Candidate model : 5/10)
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



#############################################
#### 8. Save Model
#############################################

# Output
import pickle
filename = './model/Ensemble_Model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(votingC, f)

filename = "./model/Best_Single_Model.pkl".format(best_single_model)
with open(filename, 'wb') as f:
    pickle.dump(cls_dict[best_single_model], f)


filename = "./model/Best_Model.pkl".format(best_model)
with open(filename, 'wb') as f:
    pickle.dump(cls_dict[best_model], f)
