import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

'''# Churn Prediction'''
'''#### Problem Statement:
The primary objective is to develop a churn prediction model that accurately identifies customers at risk of churning. This model will enable proactive measures to retain customers and minimize revenue loss. Specifically, the model should:

- Predict whether a customer is likely to churn within a defined future time period (e.g., next month).
- Provide insights into the key factors influencing churn, empowering the business to take targeted actions.
- Offer a scalable solution that can be integrated into existing systems for real-time or periodic churn prediction.

#### Data Description:
The dataset consists of historical customer information, including demographics, subscription details, usage patterns, interactions with the service, and any other relevant features. Key attributes may include:

- Customer demographics (age, gender, location, etc.).
- Subscription plan details (plan type, subscription duration, etc.).
- Usage behavior (frequency of use, time spent on the platform, etc.).
- Transactional data (payment history, purchase frequency, etc.).
- Customer interactions (customer support tickets, feedback, etc.).
#### Evaluation Metrics:
The performance of the churn prediction model will be assessed using relevant evaluation metrics such as accuracy, precision, recall, F1 score, and area under the ROC curve (AUC). Additionally, the model's effectiveness in reducing churn rate and retaining high-value customers will be monitored over time.

#### Deliverables:

A robust churn prediction model trained on historical data.
Documentation outlining the model's architecture, feature importance, and deployment guidelines.
Insights into customer behavior and factors driving churn.
Recommendations for targeted marketing campaigns, personalized offers, and retention strategies based on the model's predictions.'''

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

pred = None
if button:
    with open("model.pkl","rb") as f:
        model = pickle.load(f)
    pred = model.predict(pred_data)

pred

telecom_data = pd.read_csv("telecom.csv")
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

st.write(plot_stacked_bar('Partner'))