import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

st.write('''# Churn Prediction''')
st.write('''#### Problem Statement:
        The primary objective is to develop a churn prediction model that accurately identifies customers at risk of churning. This model will enable proactive measures to retain customers and minimize revenue loss. Specifically, the model should:'''
)
st.write('''- Predict whether a customer is likely to churn within a defined future time period (e.g., next month).
            - Provide insights into the key factors influencing churn, empowering the business to take targeted actions.
        - Offer a scalable solution that can be integrated into existing systems for real-time or periodic churn prediction.''')

st.write('''#### Data Description:
        The dataset consists of historical customer information, including demographics, subscription details, usage patterns, interactions with the service, and any other relevant features. Key attributes may include:

        Customer demographics (age, gender, location, etc.).
        Subscription plan details (plan type, subscription duration, etc.).
        Usage behavior (frequency of use, time spent on the platform, etc.).
        Transactional data (payment history, purchase frequency, etc.).
        Customer interactions (customer support tickets, feedback, etc.).''')
st.write('''#### Evaluation Metrics:
        The performance of the churn prediction model will be assessed using relevant evaluation metrics such as accuracy, precision, recall, F1 score, and area under the ROC curve (AUC). Additionally, the model's effectiveness in reducing churn rate and retaining high-value customers will be monitored over time.''')

st.write('''#### Deliverables:
             The goal is to obtain these results:   
        A robust churn prediction model trained on historical data.
        Documentation outlining the model's architecture, feature importance, and deployment guidelines.
        Insights into customer behavior and factors driving churn.
        Recommendations for targeted marketing campaigns, personalized offers, and retention strategies based on the model's predictions.''')


threshold = 40
telecom_data = pd.read_csv("telecom.csv")

def highlight_exceeding_value(val):
    color = 'red' if val > threshold else 'black'
    return f'color: {color}'

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


uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
if uploaded_file:
    pred_data = pd.read_csv(uploaded_file)
    button = st.button("Predict")
    if button:
        with open('encoder.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        for column, le in label_encoders.items():
            pred_data[column] = le.transform(pred_data[column])
        
        with open("model.pkl","rb") as f:
            model = pickle.load(f)
        pred = pd.DataFrame(model.predict_proba(pred_data))
        pred = pred*100
        column_names = [r"Probability % of churning",r"Probability % of not churning"]
        pred.columns = column_names
        
        styled_pred = pred.style.applymap(highlight_exceeding_value, subset=[r"Probability % of churning"])
        st.write(styled_pred)
    
st.write("#### Alternatively, you can also manually input the data below to make predictions for a single instance.")
    
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

if st.checkbox("Analysis"):

    telecom_data = pd.read_csv("telecom.csv")
    column = telecom_data.columns.drop(["customerID","TotalCharges","MonthlyCharges","tenure"])
    col_data = st.selectbox(label = "Choose below", options=column)
    graph = plot_stacked_bar(col_data)
    st.write(graph)