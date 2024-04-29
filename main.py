import streamlit as st
import analysis
import prediction
import home

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home","Prediction","Analysis"])

    if page == "Analysis":
        analysis.show()
    elif page == "Prediction":
        prediction.show()
    elif page == "Home":
        home.show()

if __name__ == "__main__":
    main()
