import streamlit as st
import analysis
import prediction

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Prediction","Analysis"])

    if page == "Analysis":
        analysis.show()
    elif page == "Prediction":
        prediction.show()

if __name__ == "__main__":
    main()
