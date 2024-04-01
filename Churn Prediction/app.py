import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("prepared_data.csv")

features = data.drop(['Churn','TotalCharges'],axis=1)
label = data['Churn']

columns = features.columns

for column in columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])

x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=47)

param_grid = {'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
clf = DecisionTreeClassifier(**param_grid)

clf.fit(x_train,y_train)

def prediction(data):
    return clf.predict(data)
