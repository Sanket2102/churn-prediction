import streamlit as st
def show():
    st.write('''# Churn Prediction''')
    st.write('''#### Problem Statement:
        The primary objective is to develop a churn prediction model that accurately identifies customers at risk of churning. This model will enable proactive measures to retain customers and minimize revenue loss. Specifically, the model should:

        - Predict whether a customer is likely to churn within a defined future time period (e.g., next month).
        - Provide insights into the key factors influencing churn, empowering the business to take targeted actions.
        - Offer a scalable solution that can be integrated into existing systems for real-time or periodic churn prediction.''')

    st.write('''#### Data Description:
        The dataset consists of historical customer information, including demographics, subscription details, usage patterns, interactions with the service, and any other relevant features. Key attributes may include:

        - Customer demographics (age, gender, location, etc.).
        - Subscription plan details (plan type, subscription duration, etc.).
        - Usage behavior (frequency of use, time spent on the platform, etc.).
        - Transactional data (payment history, purchase frequency, etc.).
        - Customer interactions (customer support tickets, feedback, etc.).''')
    st.write('''#### Evaluation Metrics:
        The performance of the churn prediction model will be assessed using relevant evaluation metrics such as accuracy, precision, recall, F1 score, and area under the ROC curve (AUC). Additionally, the model's effectiveness in reducing churn rate and retaining high-value customers will be monitored over time.''')

    st.write('''#### Deliverables:
             The goal is to obtain these results.
        -A robust churn prediction model trained on historical data.
        -Documentation outlining the model's architecture, feature importance, and deployment guidelines.
        -Insights into customer behavior and factors driving churn.
        -Recommendations for targeted marketing campaigns, personalized offers, and retention strategies based on the model's predictions.''')
