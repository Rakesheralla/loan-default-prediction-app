import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Loan Default Prediction",
    layout="centered"
)

# ---------------- TITLE ----------------
st.title("🏦 Loan Default Prediction System")

st.write("Upload a CSV file to predict loan default risk")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("model/loan_default_model.pkl")
    features = list(joblib.load("model/model_features.pkl"))
    st.success("Model loaded successfully")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV file)",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV
        data = pd.read_csv(uploaded_file)

        st.subheader("📄 Uploaded Data")
        st.write(data.head())

        # ---------------- CLEANING ----------------
        # Remove unwanted columns if present
        data = data.drop(columns=["TARGET", "SK_ID_CURR"], errors='ignore')

        # Match model features
        data = data.reindex(columns=features, fill_value=0)

        # ---------------- PREDICTION ----------------
        predictions = model.predict(data)

        # Add prediction column
        data["Prediction"] = predictions

        # Convert to readable labels
        data["Result"] = data["Prediction"].map({
            0: "Not Defaulter",
            1: "Defaulter"
        })

        # ---------------- OUTPUT ----------------
        st.subheader("📊 Prediction Results")
        st.write(data[["Prediction", "Result"]])

    except Exception as e:
        st.error(f"Error processing file: {e}")