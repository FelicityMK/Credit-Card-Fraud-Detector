import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")


st.title("Credit Card Fraud Detection App")
st.write("Upload a transaction dataset (CSV format) to check for fraud.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.error("The uploaded file is empty.")
        else:
            st.subheader("Preview of Dataset")
            st.write(df.head())

            # Define fraud and non-fraud counts
            fraud_count = df[df['Class'] == 1].shape[0]
            non_fraud_count = df[df['Class'] == 0].shape[0]

            st.success(f"Detected {fraud_count} fraudulent transactions.")
            st.info(f"Detected {non_fraud_count} non-fraudulent transactions.")

            st.subheader("Class Distribution")
            st.bar_chart(df['Class'].value_counts())

            # Prepare input features
            if "Class" in df.columns:
                input_df = df.drop("Class", axis=1)
            else:
                input_df = df.copy()

            # Scale input
            scaled_input = scaler.transform(input_df)

            # Predict
            predictions = model.predict(scaled_input)
            df["Prediction"] = predictions
            df["Prediction"] = df["Prediction"].map({0: "Legit", 1: "Fraud"})

            # Show prediction summary
            fraud_predicted = df["Prediction"].value_counts().get("Fraud", 0)
            st.subheader(" Prediction Results")
            st.write(
                df[["Prediction"]]
                .value_counts()
                .rename_axis("Result")
                .reset_index(name="Count")
            )
            st.success(
                f"The model predicts {fraud_predicted} fraudulent transactions in this dataset."
            )

            # Show full predictions table
            with st.expander("See Full Prediction Table"):
                st.write(df)

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.warning("Please upload a CSV file to continue.")


model = joblib.load("models/xgboost_model.pkl")
scaler = joblib.load("models/scaler.pkl")




