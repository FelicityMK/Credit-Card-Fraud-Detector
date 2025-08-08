import streamlit as st
import pandas as pd
import joblib
import requests
from io import BytesIO

# Set page config
st.set_page_config(page_title="Credit Card Fraud Detection", layout="centered")

# Load model and scaler
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# Cache the CSV download
@st.cache_data(show_spinner=True)
def load_default_data():
    url = "https://drive.google.com/uc?export=download&id=1kcFYu5LfKH74Qcmrf7KQLz2D7S7RRKHh"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError("Failed to download dataset.")
    data = pd.read_csv(BytesIO(response.content))
    if data.empty:
        raise ValueError("The downloaded file is empty.")
    return data

# App title and intro
st.title("Credit Card Fraud Detection App")
st.write("Upload your transaction file or use the cached dataset to predict fraudulent transactions.")

# Upload CSV or use cached
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
use_cached = st.checkbox("Use cached default dataset instead", value=False)

# Load data accordingly
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading uploaded file: {e}")
elif use_cached:
    try:
        df = load_default_data()
        st.success("Using cached dataset from Google Drive.")
    except Exception as e:
        st.error(f"Error loading cached dataset: {e}")
else:
    st.info("Upload a dataset or check the box to use the default one.")

# Run model prediction
if df is not None:
    if df.empty:
        st.error("The dataset is empty.")
    else:
        st.subheader("Raw Data Preview")
        st.write(df.head())

        # Fraud stats
        fraud_count = df[df["Class"] == 1].shape[0]
        non_fraud_count = df[df["Class"] == 0].shape[0]

        st.success(f"Detected {fraud_count} fraudulent transactions.")
        st.info(f"Detected {non_fraud_count} non-fraudulent transactions.")

        # Bar chart
        st.subheader("Class Distribution")
        st.bar_chart(df["Class"].value_counts())

        # Prepare input features
        input_df = df.drop("Class", axis=1) if "Class" in df.columns else df.copy()
        scaled_input = scaler.transform(input_df)

        # Predict
        predictions = model.predict(scaled_input)
        df["Prediction"] = predictions
        df["Prediction"] = df["Prediction"].map({0: "Legit", 1: "Fraud"})

        # Summary
        st.subheader("Prediction Results")
        st.write(
            df[["Prediction"]]
            .value_counts()
            .rename_axis("Result")
            .reset_index(name="Count")
        )
        fraud_predicted = df["Prediction"].value_counts().get("Fraud", 0)
        st.success(f"The model predicts {fraud_predicted} fraudulent transactions in this dataset.")

        # Full table
        with st.expander("See Full Prediction Table"):
            st.dataframe(df)





