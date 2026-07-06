import streamlit as st
import requests

st.title("Customer Churn Prediction")

tenure = st.number_input("Tenure", 0)
monthly = st.number_input("Monthly Charges", 0.0)
total = st.number_input("Total Charges", 0.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])

if st.button("Predict"):

    payload = {
        "tenure": tenure,
        "monthly_charges": monthly,
        "total_charges": total,
        "contract": contract,
        "internet_service": internet
    }

    response = requests.post("http://127.0.0.1:8000/predict", json=payload)

    st.write(response.json())