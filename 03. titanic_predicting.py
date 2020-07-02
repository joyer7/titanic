#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# Model Predicting
# 예측결과를 MongDB, KUDU에 저장
#-------------------------------------------------------

# Import the basic python libraries
import pandas as pd
import pickle
import os
import warnings
import numpy as np
from impala.dbapi import connect
from impala.util import as_pandas
import datetime
warnings.filterwarnings('ignore')

## set dir
#os.chdir('/home/cdsw/titanic_example')

## Load data
IMPALA_HOST = os.getenv('IMPALA_HOST','45.xxx.xxx.xxx')
conn = connect(host=IMPALA_HOST, port=21050, \
               user='xxxxx', password='xxxxxxx', \
               use_ssl=False, \
               auth_mechanism='LDAP')


cursor = conn.cursor(user='xxxxxx')
cursor.execute('use titanic')
cursor.execute('select * from TITANIC_DATA')
combdata = as_pandas(cursor)
IDtest = pd.DataFrame()
IDtest['PassengerId'] = combdata.loc[combdata.kind =='test', 'passengerid']
IDtest.reset_index(inplace=True)
IDtest.drop('index',1,inplace=True)
del combdata

test = pd.read_csv("./prep_data/prep_test.csv")
now_datetime = datetime.datetime.now()

#############################################
#### 8. Load Model & Predict
#############################################
## Load model
filename = './model/Ensemble_Model.pkl'
with open(filename, 'rb') as f:
    ens_model = pickle.load(f)

filename = "./model/Best_Single_Model.pkl"
with open(filename, 'rb') as f:
    best_single_model = pickle.load(f)

filename = "./model/Best_Model.pkl"
with open(filename, 'rb') as f:
    best_model = pickle.load(f)


ens_pred = pd.Series(ens_model.predict(test), name="Survived")
ens_pred_df = pd.concat([IDtest,ens_pred],axis=1)
#ens_pred_df.to_csv("outpred/Ensemble_Model_Predict.csv",index=False)
ens_pred_df['MODEL'] = 'Ensemble_Model'

best_single_pred = pd.Series(best_single_model.predict(test), name="Survived")
best_single_pred_df = pd.concat([IDtest,best_single_pred],axis=1)
#best_single_pred_df.to_csv("outpred/Best_Single_Predict.csv",index=False)
best_single_pred_df['MODEL'] = 'Best_Single_Model'

best_pred = pd.Series(best_model.predict(test), name="Survived")
best_pred_df = pd.concat([IDtest,best_pred],axis=1)
#best_pred_df.to_csv("outpred/Best_Predict.csv",index=False)
best_pred_df['MODEL'] = 'Best_Model'

# 
all_result = pd.concat([ens_pred_df, best_single_pred_df, best_pred_df],ignore_index=True)
ndtime = now_datetime.strftime("%Y-%m-%d %H:%M:%S")
all_result['predictdatetime'] = ndtime
all_result = all_result[['predictdatetime', 'PassengerId', 'MODEL', 'Survived']]
all_result[all_result.select_dtypes(include=['object']).columns] = all_result.select_dtypes(include=['object']).astype(str)



#############################################
#### 9. Save predict result 
#############################################
#----------------------
## A) Save CSV
all_result.to_csv("outpred/[{}]_predict_all.csv".format(ndtime), index=False)
help(all_result.to_csv)

#----------------------
## B) Save kudu table
from impala.dbapi import connect
from impala.util import as_pandas

# B-1) Make connection
IMPALA_HOST = os.getenv('IMPALA_HOST','xxx.xxx.xxx.xxx')
print(IMPALA_HOST)

conn = connect(host=IMPALA_HOST, port=21050, \
               user='xxxxxx', password='xxxxxxxxx', \
               use_ssl=False, \
               auth_mechanism='LDAP')

cursor = conn.cursor(user='xxxxx')
cursor.execute('use default')

# B-2) Insert result
def input_kudu(df, table):
    sql = ''
    for i, row in enumerate(df.values):
        mrow = [-1 if ((type(e) ==float) and (pd.isna(e))) else e for e in row]
        cursor.execute(' '.join(['insert into', table, 'values', str(tuple(mrow))]))
input_kudu(all_result, 'TITANIC_RESULT')
cursor.execute('select * from TITANIC_RESULT')

# B-3) Check insert result 
tables = as_pandas(cursor)
tables

# B-4) Close connect
cursor.close()
conn.close()


#----------------------
## C) Save MongoDB
import pymongo
import datetime

# C-1) Make connection
username = 'xxxxx'
password = 'xxxxxx'
connection = pymongo.MongoClient('mongodb://%s:%s@45.xxx.xxx.xxx'%(username,password))
db = connection["cdsw"]
titanic_collection = db["titanic"]


# C-2) Make data(to save mongoDB)
#  C-2-a) make datetime
now_datetime = datetime.datetime.now()
ndtime = now_datetime.strftime("%Y-%m-%d %H:%M:%S")
ndate = now_datetime.strftime("%Y-%m-%d")
nhour = now_datetime.strftime("%H")
nmin = now_datetime.strftime("%M")
nsec = now_datetime.strftime("%S")
data = {'datetime':ndtime,
'date':ndate,
'hour':nhour,
'minute':nmin,
'second':nsec}


#  C-2-b) make key:value struct
all_result['PassengerId'] = all_result['PassengerId'].astype(str)
all_result['Survived'] = all_result['Survived'].astype(str)
all_result.drop('predictdatetime', 1)
ens_m = all_result.loc[all_result.MODEL == 'Ensemble_Model',['PassengerId', 'Survived']]
best_single_m = all_result.loc[all_result.MODEL == 'Best_Single_Model',['PassengerId', 'Survived']]
best_m = all_result.loc[all_result.MODEL == 'Best_Model',['PassengerId', 'Survived']]


def set_index(dfi, idx):
    # 입력 dataframe과 index로 사용할 컬럼
    # 반환 index 수정 및 사용한 index 컬럼 제거한 dataframe
    df = dfi.copy()
    df.index = df[idx].values
    return df.drop(idx, axis=1)
predict_json = {'PredictResult':dict()}
predict_json['PredictResult']['Ensemble_Model'] = set_index(ens_m,'PassengerId').T.to_dict()
predict_json['PredictResult']['Best_Single_Model'] = set_index(best_single_m,'PassengerId').T.to_dict()
predict_json['PredictResult']['Best_Model'] = set_index(best_m,'PassengerId').T.to_dict()
data = dict(data, **predict_json)


print(data)
# C-3) Insert result
titanic_collection.insert_one(data)
titanic_collection

# C-4) Check insert result 
def make_decision(sltd_doc, PassengerId, model_list = ['Ensemble_Model', 'Best_Single_Model', 'Best_Model']):
    # 입력 : 모델 예측 결과가 저장 된 sltd_doc, PassengerId
    # 출력 : BestSingle, Best, Ensemble 세가지 모델 다수결 예측 결과 반환
    decision = ['die', 'survival']
    model_predicts = [int(sltd_doc['PredictResult'][model][str(PassengerId)]['Survived']) \
                      for model in model_list]
    predict = round(sum(model_predicts)/len(model_predicts))
    print(' PassengerId : {} -> {}'.format(PassengerId, decision[predict]))

#  C-4-a)load saved model(datetime ref: C-2)
sltd_docs = [doc for doc in titanic_collection.find({'datetime':ndtime})]
print(len(sltd_docs))
sltd_doc = sltd_docs[0]

#  C-4-b)predict some of passenger
for pid in [1111, 1112, 1113]:
    make_decision(sltd_doc, pid)

# B-4) Close connect
connection.close()