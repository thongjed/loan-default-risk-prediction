import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn 

# Load model and feature names
with open('FYP_Data_Preprocessing/loan_approval_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('FYP_Data_Preprocessing/model_features.pkl', 'rb') as file:
    feature_names = pickle.load(file)

# App title
st.title("Loan Approval Prediction")

# Input form
st.subheader("Enter applicant details:")

# Dynamically create input fields
user_input = {}
for feature in feature_names:
    value = st.text_input(f"{feature}")
    user_input[feature] = value

# Predict button
if st.button("Predict"):
    try:
        # Convert to DataFrame and float
        input_df = pd.DataFrame([user_input])
        input_df = input_df.astype(float)

        # Predict
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][int(prediction)]

        st.success(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'} (Confidence: {prediction_proba:.2f})")
    except Exception as e:
        st.error(f"Error: {e}")

