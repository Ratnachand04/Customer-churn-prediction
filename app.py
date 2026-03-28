import streamlit as st
import pandas as pd
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="👥",
    layout="wide"
)

st.title("Customer Churn Prediction Dashboard")
st.write("Welcome to the Customer Churn Prediction App. In the future, this app will host the machine learning model to predict customer churn.")

# Placeholder for data loading
st.subheader("Dataset Preview")
data_path = "data/raw/train.csv"
if os.path.exists(data_path):
    try:
        df = pd.read_csv(data_path, nrows=5)
        st.dataframe(df)
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
else:
    st.info("Training data not found in `data/raw/train.csv`. Please populate the required files.")
