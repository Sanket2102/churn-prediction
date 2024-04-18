import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def show():
    left_column, middle_column, right_column = st.columns(3)

    SeniorCitizen = left_column.selectbox("Do you come under senior citizen category",("Yes","No"))
    Partner = middle_column.selectbox(label="Do you have a partner",options=['Yes','No'])
    Dependents = right_column.selectbox("Do you have any dependents?",("Yes","No"))

    left_column1, middle_column1, right_column1 = st.columns(3)
    InternetService = left_column1.selectbox("Choose your Internet Service type",("DSL","Fiber optic", "No"))
    OnlineSecurity = middle_column1.selectbox("Online Security",("Yes","No"))
    OnlineBackup = right_column1.selectbox("Online Backup",("Yes","No"))

    left_column2, middle_column2, right_column2,right_column3 = st.columns(4)
    DeviceProtection = left_column2.selectbox("Device Protection",("Yes","No"))
    TechSupport = middle_column2.selectbox("Tech Support", ("Yes","No"))
    StreamingTV = right_column2.selectbox("Streaming TV",("Yes","No"))
    StreamingMovies = right_column3.selectbox("Streaming Movies", ("Yes","No"))

    column1,column2,column3 = st.columns(3)
    Contract = column1.radio("Contract",("Month-to-month","One year","Two year"))
    PaperlessBilling = column2.radio("Paperless Billing", ("Yes","No"))
    PaymentMethod = column3.radio("Payment Method", ("Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"))

    button = st.button("Submit")


    data = {
        'SeniorCitizen': [SeniorCitizen],
        'Partner': [Partner],
        'Dependents': [Dependents],
        'InternetService': [InternetService],
        'OnlineSecurity': [OnlineSecurity],
        'OnlineBackup': [OnlineBackup],
        'DeviceProtection': [DeviceProtection],
        'TechSupport': [TechSupport],
        'StreamingTV': [StreamingTV],
        'StreamingMovies': [StreamingMovies],
        'Contract': [Contract],
        'PaperlessBilling': [PaperlessBilling],
        'PaymentMethod': [PaymentMethod]
    }

    pred_data = pd.DataFrame(data)
    st.write(pred_data)

    with open('encoder.pkl', 'rb') as f:
        label_encoders = pickle.load(f)

    for column, le in label_encoders.items():
        pred_data[column] = le.transform(pred_data[column])
    
    if button:
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        pred = model.predict_proba(pred_data)
        pred = pred[0,0]
        st.write(f"The chance of this customer to churn is {(round(pred,4))*100}%")