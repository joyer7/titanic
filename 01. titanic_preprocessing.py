#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# Data Processing
#-------------------------------------------------------

#############################################
#### 1. Process
#############################################

"""
설명

"""
 
#############################################
#### 2. Read data & Import library
#############################################
# Import the basic python libraries
import pandas as pd
import numpy as np
import warnings
from impala.dbapi import connect
from impala.util import as_pandas

warnings.filterwarnings('ignore')

import os
#set dir
os.chdir('/home/cdsw/titanic_example')

IMPALA_HOST = os.getenv('IMPALA_HOST','45.xxx.xxx.xxx')
conn = connect(host=IMPALA_HOST, port=21050, \
               user='xxxxx', password='xxxxxxxx', \
               use_ssl=False, \
               auth_mechanism='LDAP')


cursor = conn.cursor(user='xxxx')
cursor.execute('use titanic')
cursor.execute('select * from TITANIC_DATA')
combdata = as_pandas(cursor)
column_match = {'passengerid':'PassengerId',
'survived':'Survived',
'pclass':'Pclass',
'name':'Name',
'sex':'Sex',
'age':'Age',
'sibsp':'SibSp',
'parch':'Parch',
'ticket':'Ticket',
'fare':'Fare',
'cabin':'Cabin',
'embarked':'Embarked',
'kind':'kind'}
combdata = combdata.rename(columns = column_match)

for c  in combdata.columns:
  print(combdata[c].dtype)

dtypelist = [int, int, int,str, str,float,int,int,str,float,str, str,str]
for i, c in enumerate(combdata.columns):
  combdata[c] = combdata[c].astype(dtypelist[i])
  
combdata = combdata.replace(-1, np.nan)
combdata = combdata.replace('nan', np.nan)  
combdata.info()
#############################################
#### 3. Exploratory data analysis
#############################################
# Skip


#############################################
#### 4. Data preparation including feature engineering
#############################################

# Let's combining train & test for quick feature engineering. 
# Variable source is a kind of tag which indicates data source in combined data
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

## Separate train dataset and test dataset using the index variable 'kind'

train = combdata.loc[combdata['kind']=="train"]
test = combdata.loc[combdata['kind']=="test"]
test.drop(labels=["Survived"],axis = 1,inplace=True)
train.drop(labels=["kind"],axis = 1,inplace=True)
test.drop(labels=["kind"],axis = 1,inplace=True)


## Separate train features and label 

train["Survived"] = train["Survived"].astype(int)



train.to_csv("./prep_data/prep_train.csv", index=False)
test.to_csv("./prep_data/prep_test.csv", index=False)