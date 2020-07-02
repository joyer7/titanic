#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# Rest API 코드
#-------------------------------------------------------

# Import the basic python libraries
import pandas as pd
import pickle
import os
import warnings
import numpy as np
warnings.filterwarnings('ignore')


def predict(input_value, filename='./model/Ensemble_Model.pkl'):
    input_val_type = {'PassengerId' : int,
    'Pclass' : int ,
    'Name' : str ,
    'Sex' : str ,
    'Age' : float ,
    'SibSp' : int ,
    'Parch' : int ,
    'Ticket' : str ,
    'Fare' : float ,
    'Cabin' : str ,
    'Embarked' : str}

    
    convt_input_val = {}
    for col in input_val_type.keys():
        convt_input_val[col] = input_val_type[col](input_value[col])

    test = pd.DataFrame(convt_input_val, index=[0])

    convt_input_val
    test.drop(labels = ["PassengerId"], axis = 1, inplace = True)

    # Pclass 
    test['Pclass'].unique()

    # Name - Extract Salutation from Name variable
    salutation = [i.split(",")[1].split(".")[0].strip() for i in test["Name"]]
    test["Title"] = pd.Series(salutation)
    test["Title"].unique()

    # Convert other salutations to fixed Title 
    test["Title"] = test["Title"].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    test["Title"] = test["Title"].map({"Master":0, "Miss":1, "Ms" : 1 , "Mme":1, "Mlle":1, "Mrs":1, "Mr":2, "Rare":3})
    test["Title"] = test["Title"].astype(int)
    test["Title"].unique()

    test = pd.get_dummies(test, columns = ["Title"])

    # Drop Name variable
    test.drop(labels = ["Name"], axis = 1, inplace = True)


    # Age
    ## Fill Age with the median age of similar rows according to Sex, Pclass, Parch and SibSp
    # Index of NaN age rows
    missing_index = list(test["Age"][test["Age"].isnull()].index)

    for i in missing_index :
        median_age = test["Age"].median()
        filled_age = test["Age"][((test['Sex'] == test.iloc[i]["Sex"]) & 
                                    (test['SibSp'] == test.iloc[i]["SibSp"]) & 
                                    (test['Parch'] == test.iloc[i]["Parch"]) & 
                                    (test['Pclass'] == test.iloc[i]["Pclass"]))].median()
        if not np.isnan(filled_age) :
            test['Age'].iloc[i] = filled_age
        else :
            test['Age'].iloc[i] = median_age



    # Sex - Create dummy variables
    #test["Sex"] = test["Sex"].map({"male": 0, "female":1}) or
    test = pd.get_dummies(test, columns = ["Sex"])  

    # Create a variable representing family size from SibSp and Parch
    test["Fsize"] = test["SibSp"] + test["Parch"] + 1

    # Create new feature of family size
    test['Single'] = test['Fsize'].map(lambda s: 1 if s == 1 else 0)
    test['SmallF'] = test['Fsize'].map(lambda s: 1 if  s == 2  else 0)
    test['MedF'] = test['Fsize'].map(lambda s: 1 if 3 <= s <= 4 else 0)
    test['LargeF'] = test['Fsize'].map(lambda s: 1 if s >= 5 else 0)

    # Drop Name variable
    test.drop(labels = ["Fsize"], axis = 1, inplace = True)

    # SibSp - Create dummy variables
    test = pd.get_dummies(test, columns = ["SibSp"])

    # Parch - Create dummy variables
    test = pd.get_dummies(test, columns = ["Parch"])

    # Ticket - Extracting the ticket prefix. This might be a representation of class/compartment.
    # If there is no prefix replace with U (Unknown). 

    Ticket = []
    for i in list(test.Ticket):
        if not i.isdigit() :
            Ticket.append(i.replace(".","").replace("/","").strip().split(' ')[0])
        else:
            Ticket.append("U")

    test["Ticket"] = Ticket
    test["Ticket"].unique()

    test = pd.get_dummies(test, columns = ["Ticket"], prefix="T")

    # Fare - Check the number of missing value
    test["Fare"].isnull().sum()

    # Only 1 value is missing so we will fill the same with median
    test["Fare"] = test["Fare"].fillna(test["Fare"].median())

    # Use the numpy fuction log1p which  applies log(1+x) to all elements of the column
    test["Fare"] = np.log1p(test["Fare"])


    # Cabin - Replace the missing Cabin number by the type of cabin unknown 'U'
    test["Cabin"].fillna('U', inplace=True)

    # Create dummy variables
    test = pd.get_dummies(test, columns = ["Cabin"], prefix="Cabin")

    # Embarked - Find the number of missing values
    test["Embarked"].isnull().sum()

    # Fill Embarked missing values of dataset set with mode 'S'
    test["Embarked"] = test["Embarked"].fillna("S")

    # Create dummy variables
    test = pd.get_dummies(test, columns = ["Embarked"], prefix="Emb")

    # Create dummies for PClass Now
    test = pd.get_dummies(test, columns = ["Pclass"], prefix="Pclass")

    # Compare Train<-> Test variables
    test_col = list(test.columns)
    train_col = ['Age', 'Fare', 'Title_0', 'Title_1', 'Title_2', 'Title_3', 'Sex_female',
       'Sex_male', 'Single', 'SmallF', 'MedF', 'LargeF', 'SibSp_0', 'SibSp_1',
       'SibSp_2', 'SibSp_3', 'SibSp_4', 'SibSp_5', 'SibSp_8', 'Parch_0',
       'Parch_1', 'Parch_2', 'Parch_3', 'Parch_4', 'Parch_5', 'Parch_6',
       'Parch_9', 'T_A', 'T_A4', 'T_A5', 'T_AQ3', 'T_AQ4', 'T_AS', 'T_C',
       'T_CA', 'T_CASOTON', 'T_FC', 'T_FCC', 'T_Fa', 'T_LINE', 'T_LP', 'T_PC',
       'T_PP', 'T_PPP', 'T_SC', 'T_SCA3', 'T_SCA4', 'T_SCAH', 'T_SCOW',
       'T_SCPARIS', 'T_SCParis', 'T_SOC', 'T_SOP', 'T_SOPP', 'T_SOTONO2',
       'T_SOTONOQ', 'T_SP', 'T_STONO', 'T_STONO2', 'T_STONOQ', 'T_SWPP', 'T_U',
       'T_WC', 'T_WEP', 'Cabin_A', 'Cabin_B', 'Cabin_C', 'Cabin_D', 'Cabin_E',
       'Cabin_F', 'Cabin_G', 'Cabin_T', 'Cabin_U', 'Emb_C', 'Emb_Q', 'Emb_S',
       'Pclass_1', 'Pclass_2', 'Pclass_3']

    null_col_list =[trc for trc in train_col if trc not in test_col]
    new_col_list =[trc for trc in test_col if trc not in train_col]
    for null_col in null_col_list:
        test[null_col] = np.zeros(len(test))
    test.drop(labels = new_col_list, axis = 1, inplace = True)    


    #filename : model/Ensemble_Model.pkl | model/[Best_Single_Model]_{}.pkl | model/[Best_Model]_{}.pkl
    with open(filename, 'rb') as f:
        model = pickle.load(f)
    return model.predict(test)[0]

  

#test = {
#  "PassengerId": "2",
#  "Pclass": "1",
#  "Name": "Cumings, Mrs. John Bradley (Florence Briggs Th...",
#  "Sex": "female",
#  "Age": "38",
#  "SibSp": "1",
#  "Parch": "0",
#  "Ticket": "PC 17599",
#  "Fare": "71.2833",
#  "Cabin": "C85",
#  "Embarked": "C"
#}
#
#
#test2 = {'PassengerId' : '892',
#'Pclass' : '3',
#'Name' : 'Kelly,Mr.James' ,
#'Sex' : 'male' ,
#'Age' : '34.5' ,
#'SibSp' : '0' ,
#'Parch' : '0' ,
#'Ticket' : '330911' ,
#'Fare' : '7.8292' ,
#'Cabin' : '' ,
#'Embarked' : 'Q'}
#  
#predict(test)

print("Hello")