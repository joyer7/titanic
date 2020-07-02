#-------------------------------------------------------
# Author : SHIN SEOK RHO
#
# kudu table data insert (from csv)
#
#-------------------------------------------------------
####################
## Train + Test Dataset
####################
import os
import pandas as pd
from impala.dbapi import connect
from impala.util import as_pandas
import os

#set dir
#os.chdir('/home/cdsw/titanic_example')

#Read the datasets
train = pd.read_csv("./data/titanic_train.csv")
test = pd.read_csv("./data/titanic_test.csv")
train['kind'] = 'train'
test['kind'] = 'test'
test['Survived'] = -1
train['Survived'] =  train['Survived']
combdata = pd.concat([train, test],ignore_index=True).loc[:,train.columns]

for col in combdata.columns:
  combdata[col] = combdata[col].astype(train[col].dtype)

combdata[combdata.select_dtypes(include=['object']).columns] = combdata.select_dtypes(include=['object']).astype(str)


IMPALA_HOST = os.getenv('IMPALA_HOST','45.xxx.xxx.xxx')
print(IMPALA_HOST)
#conn = connect(host=IMPALA_HOST, port=21050, user='htd4973', password='Epdlxjqnstjr20!', use_ssl=False)
conn = connect(host=IMPALA_HOST, port=21050, \
               user='xxxxxx', password='xxxxxxxxxx!', \
               use_ssl=False, \
               auth_mechanism='LDAP')


cursor = conn.cursor(user='xxxxxx')

cursor.execute('use default')

table = 'TITANIC_DATA'
sql = ''
for i, row in enumerate(combdata.values):
  mrow = [-1 if ((type(e) ==float) and (pd.isnull(e))) else e for e in row]
  cursor.execute(' '.join(['insert into', table, 'values', str(tuple(mrow))]))
  
cursor.execute('select * from TITANIC_DATA')
tables = as_pandas(cursor)
tables

