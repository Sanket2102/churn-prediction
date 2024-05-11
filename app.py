#importing essential libraries
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# textual summary of the project for the webapp header
st.write('''# Churn Prediction''')
st.write('''#### Problem Statement:''')
st.image("image.png")
st.write('''The primary objective is to develop a churn prediction model that accurately identifies customers at risk of churning. This model will enable proactive measures to retain customers and minimize revenue loss. Specifically, the model should:
         
        1. Predict whether a customer is likely to churn within a defined future time period (e.g., next month).
        2. Provide insights into the key factors influencing churn, empowering the business to take targeted actions.
        3. Offer a scalable solution that can be integrated into existing systems for real-time or periodic churn prediction.''')

st.write('''#### Data Description:''')
st.write(
    '''The dataset consists of historical customer information, including demographics, subscription details, usage patterns, interactions with the service, and any other relevant features. Key attributes include:

        1. Customer demographics (age, gender, dependants, etc.).
        2. Subscription plan details (plan type, subscription duration, etc.).
        3. Transactional data (payment history, purchase frequency, etc.).'''
         )
st.write('''#### Evaluation Metrics:''')
st.write('''The performance of the churn prediction model was assessed using accuracy score.''')

st.write('''#### Deliverables:''')
st.write(''' The goal was to obtain these results:   
        
         A robust churn prediction model trained on historical data.
        Documentation outlining the model's architecture, feature importance, and deployment guidelines.
        Insights into customer behavior and factors driving churn.
        Recommendations for targeted marketing campaigns, personalized offers, and retention strategies based on the model's predictions.'''
         )

# loading the dataset 
telecom_data = pd.read_csv("telecom.csv")

# threshold defined to mark data above a certain probability of churning
threshold = 40

# function to highlight values that are above threshold
def highlight_exceeding_value(val):
    color = 'red' if val > threshold else 'black'
    return f'color: {color}'

# function to plot bar charts for analysis
def plot_stacked_bar(column,data=telecom_data):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Plot 1: Stacked bar chart with counts
    axs[0].set_title(f'Churn Distribution by {column} (Counts)')
    axs[0].set_xlabel(f'{column}')
    axs[0].set_ylabel('Count')

    # Plotting the bar chart
    churn_counts = data.groupby([f'{column}', 'Churn']).size().unstack(fill_value=0)
    bars_counts = churn_counts.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=axs[0])

    # Add total counts on top of each bar in the first plot
    for p in bars_counts.patches:
        height = p.get_height()
        width = p.get_width()
        x, y = p.get_xy()
        axs[0].text(x + width / 2, y + height / 2, f'{int(height)}', ha='center', va='center', fontweight='bold', color='white')

    # Plot 2: Stacked bar chart with percentages
    axs[1].set_title(f'Churn Distribution by {column} (Percentages)')
    axs[1].set_xlabel(f'{column}')
    axs[1].set_ylabel('Percentage')

    # Plotting the bar chart with percentages
    churn_percentages = churn_counts.div(churn_counts.sum(axis=1), axis=0) * 100
    bars_percentages = churn_percentages.plot(kind='bar', stacked=True, color=['#ff9999', '#66b3ff'], ax=axs[1], alpha=0.5, legend=False)

    # Add percentage labels on top of each bar in the second plot
    for p in bars_percentages.patches:
        height = p.get_height()
        width = p.get_width()
        x, y = p.get_xy()
        axs[1].text(x + width / 2, y + height / 2, f'{height:.1f}%', ha='center', va='center', fontweight='bold', color='black')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    return fig

# added expander element for the user to download test datasets.
with st.expander("To download test data for checking the model, click here"):
    # drive link for downloading test dataset
    st.write("Download here- https://drive.google.com/drive/u/2/folders/1p_I4cHCl6jBU_5MDvAOPx3uhsBmwhJbO")

# this stores uploaded csv file for prediction in a separate variable
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")


if uploaded_file:
    # loads the data
    pred_data = pd.read_csv(uploaded_file)
    button = st.button("Predict")
    if button:

        # loads the encoder for label encoding of the features
        with open('encoder.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        # encodes every column into binary data
        for column, le in label_encoders.items():
            pred_data[column] = le.transform(pred_data[column])
        
        #loads trained model for prediction
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        
        # using the trained model to predict probability of churn on new data
        pred_proba = pd.DataFrame(model.predict_proba(pred_data))

        # converts data into range of 0-100 from 0-1 for better understanding
        pred_proba = pred_proba*100

        # set the column names 
        column_names = [r"Probability % of churning",r"Probability % of not churning"]
        pred_proba.columns = column_names
        
        # marks the data where probability of churning is higher using the "highlight_exceeding_value" function which we defined earlier
        styled_pred = pred_proba.style.applymap(highlight_exceeding_value, subset=[r"Probability % of churning"])
        st.write(styled_pred)
    
st.write("#### Alternatively, you can also manually input the data below to make predictions for a single instance.")
    
# For manually entering data. Used for a single instance prediction.
# created several selectboxes to fill categorical data.
if st.checkbox("Enter data manually for prediction"):
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

    button = st.button("Predict")

    # putting together all the separate data inputs in a single dictionary to make dataframe later.
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
    # loads the data in a pandas dataframe
    pred_data = pd.DataFrame(data)
    # shows the input data to the user
    st.write(pred_data)

    # loads the encoder for label encoding of the features
    with open('encoder.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    # encodes every column into binary data
    for column, le in label_encoders.items():
        pred_data[column] = le.transform(pred_data[column])
        
    if button:
        #loads trained model for prediction
        with open("model.pkl","rb") as f:
            model = pickle.load(f)

        # using the trained model to predict probability of churn on new data
        pred = model.predict_proba(pred_data)

        # fetching the probability of churning
        pred = pred[0,0]

        # displays the probability of churning in a range of 1-100
        st.write(f"The chance of this customer to churn is {(round(pred,4))*100}%")

# checkbox for analysis
if st.checkbox("Analysis"):
    # loads the data for analysis
    telecom_data = pd.read_csv("telecom.csv")

    # drops certain features which are either not categorical or irrelevant for analysis
    column = telecom_data.columns.drop(["customerID","TotalCharges","MonthlyCharges","tenure","Churn"])
    
    # selectbox to choose the feature for desired analysis
    col_data = st.selectbox(label = "Choose below", options=column)
    
    # storing the graph object created using the "plot_stacked_bar" function we defined earlier in a separate variable
    graph = plot_stacked_bar(col_data)
    # show the graph object to the user
    st.write(graph)