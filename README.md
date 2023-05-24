# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import standard libraries in python for finding Decision tree classsifier model for predicting employee churn.
2. Initialize and print the Data.head(),data.info(),data.isnull().sum()
3. Visualize data value count.
4. Import sklearn from LabelEncoder.
5. Split data into training and testing.
6. Calculate the accuracy, data prediction by importing the required modules from sklearn

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: 
RegisterNumber:  
*/


import pandas as pd
data=pd.read_csv("/content/Employee.csv")

print("data.head():")
data.head()

print("data.info():")
data.info()

print("isnull() and sum():")
data.isnull().sum()

print("data value counts():")
data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()

print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### data.head()
![238831862-b1162149-bbea-43a7-96a9-354fd108a151](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/dc7afb7c-a938-458c-97c9-40561c920bd4)
### data.info()
![238831958-bfe60847-ed9a-487c-a700-b43dca8659a5](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/4df8e1fd-c56c-4ce6-998a-fd3d0a1d11a8)
### isnull() and sum()
![238832170-7f03738f-ea1f-4338-a98b-1ef247f52708](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/69bf6431-822a-4d53-b53c-0398f0a6980f)
### data value counts()
![238834981-a4e857d7-6fa3-4dc1-b615-ec4f8433c511](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/9ff5d478-3bbf-4e48-93a0-c18d610bcc31)
### data.head() for salary
![238832413-ff85899e-0a75-4138-9b2a-d9abcdf20138](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/2872c1c0-6a53-4cea-b1aa-4a8292065f3b)
### x.head()
![238832633-6bf35c61-a89f-4af3-bdaf-840d8b320af5](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/a3041385-cd7d-4a54-8e5d-413d8ce9349a)
### accuracy value
![238832895-3010e840-d901-478a-ba93-82eba77e27e9](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/92915fac-786a-4d7b-b352-cfa81fab093e)
### data prediction
![238833279-ebb42f60-6046-426c-9759-e629b2a68b2c](https://github.com/Pravinrajj/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/117917674/1aa01be5-a09c-4e02-8b56-72331a4bdb84)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
