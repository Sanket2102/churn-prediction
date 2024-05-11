# importing essential libraries
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

# loading the dataset for model training
data = pd.read_csv("prepared_data.csv")
features = data.drop(['Churn','TotalCharges'],axis=1)
label = data['Churn']

# created empty dictionary for storing encoder objects
label_encoders = {}

# encodes each column into binary data
for column in features.columns:
    le = LabelEncoder()
    features[column] = le.fit_transform(features[column])
    label_encoders[column] = le

# Dump the encoder into a pickle file for future uses
with open("encoder.pkl","wb") as f:
    pickle.dump(label_encoders,f)

# splits the data into training and testing sets
x_train,x_test,y_train,y_test = train_test_split(features,label,test_size=0.2,random_state=47)

# best parameters for the classifier
param_grid = {'criterion': 'gini', 'max_depth': None, 'max_features': None, 'max_leaf_nodes': 10, 'min_samples_leaf': 1, 'min_samples_split': 2}

# making a decision tree classifier model with selected parameters
clf = DecisionTreeClassifier(**param_grid)

# train the model
model = clf.fit(x_train,y_train)

with open("model.pkl", "wb" ) as f:
    pickle.dump(model,f)

def prediction(data):
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    
    return model.predict(data)
