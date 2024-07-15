# CODSOFT
Internship Task 1
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

#importing data
df=pd.read_csv("D:\\dataset\\codsoft\\dataset1\\Titanic-Dataset.csv")
df.head(20)
print("Total no of passengers:"+str(len(df)))

#Analyzing data
sns.countplot(x="Survived",data=df)
sns.countplot(x="Survived",hue="Sex",data=df)
sns.countplot(x="Survived",hue="Pclass",data=df)
sns.countplot(x="Age",hue="Sex",data=df)
sns.countplot(x="Survived",hue="Pclass",data=df,palette='pastel')
df['Age'].plot.hist(color='red',figsize=(6,5))
df['Fare'].plot.hist(figsize=(6,5))
df.info()
sns.countplot(x="SibSp",data=df)

#Data wrangling
df.isnull()
df.isnull().sum()
sns.heatmap(df.isnull(),cmap='viridis')
sns.boxplot(x='Pclass',y='Age',data=df)
df.dropna
df.drop("Cabin",axis=1,inplace=True)
df.dropna(inplace=True)
sns.heatmap(df.isnull())
df.isnull().sum()
Sex=pd.get_dummies(df['Sex'],drop_first=True)
Embark=pd.get_dummies(df['Embarked'],drop_first=True)
PCLS=pd.get_dummies(df['Pclass'],drop_first=True)
df=pd.concat([df,Sex,Embark,PCLS],axis=1)
df=df.replace({True: 1, False: 0})
df.drop(['Sex','Embarked','Parch','Pclass','Name','Ticket'],axis=1,inplace=True)

Train and Test
x=df.drop('Survived',axis=1)
y=df["Survived"]
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)
from sklearn.linear_model import LogisticRegression
logmod=LogisticRegression(max_iter=1000,solver='newton-cg')
from sklearn.preprocessing import StandardScaler
​
# Assuming X is your feature matrix
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
from sklearn.preprocessing import StandardScaler
​
# Assuming X is your feature matrix
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train.columns=x_train.columns.astype(str)
logmod.fit(x_train,y_train)
LogisticRegression(max_iter=1000, solver='newton-cg')
In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.
x_test.columns=x_test.columns.astype(str)
prediction=logmod.predict(x_test)
from sklearn.metrics import classification_report
classification_report(y_test,prediction)
'              precision    recall  f1-score   support\n\n           0       0.81      0.83      0.82       126\n           1       0.74      0.72      0.73        88\n\n    accuracy                           0.78       214\n   macro avg       0.77      0.77      0.77       214\nweighted avg       0.78      0.78      0.78       214\n'
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,prediction)
array([[104,  22],
       [ 25,  63]], dtype=int64)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,prediction)*100
78.03738317757009
​
