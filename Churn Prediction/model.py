import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv("prepared_data.csv")
features = data.drop(['Churn','TotalCharges'],axis=1)
label = data['Churn']

label_encoders = {}

for column in features.columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

with open("encoder.pkl","wb") as f:
    pickle.dump(label_encoders,f)

x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=47)

param_grid = {'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}
clf = DecisionTreeClassifier(**param_grid)

clf.fit(x_train,y_train)
print(x_train.head(1))
def prediction(data):
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    
    return model.predict(data)
