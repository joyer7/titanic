import requests

url = 'https://modelservice.dcdsw3.ss.local/model'
# 생존자 Data =====================================
data1 = \
'{"accessKey":"mbh99yrjjbq4rprxoiuwult341axnpjp",\
"request":{ \
    "PassengerId":"2", \
    "Pclass":"1", \
    "Name":"Cumings, Mrs. John Bradley (Florence Briggs Th...", \
    "Sex":"female", \
    "Age":"38", \
    "SibSp":"1", \
    "Parch":"0", \
    "Ticket":"PC 17599", \
    "Fare":"71.2833", \
    "Cabin":"C85", \
    "Embarked":"C"} \
}'

# 사망자 Data =====================================
data2 = \
'{"accessKey":"mbh99yrjjbq4rprxoiuwult341axnpjp",\
"request": {\
    "PassengerId" : "892",\
    "Pclass" : "3",\
    "Name" : "Kelly,Mr.James",\
    "Sex" : "male",\
    "Age" : "34.5",\
    "SibSp" : "0",\
    "Parch" : "0",\
    "Ticket" : "330911",\
    "Fare" : "7.8292",\
    "Cabin" : "",\
    "Embarked" : "Q"} \
}'


# Request to Model =====================================
headers = {'Content-Type': 'application/json'}
response = requests.post(url, data = data2, headers= headers)
dict_json = response.json()

if dict_json['response'] == 1:
    result = "생존"
else:
    result = "사망"
    
print("\n\n\n※ Result of Titanic \n",
      "==========================================\n",
      "*" , result,
      "\n", dict_json, 
      "\n ==========================================")