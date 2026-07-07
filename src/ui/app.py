import streamlit as st
import requests

st.title("📊 Telco Churn Prediction App")

st.write("Enter customer details:")

# Inputs
payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 800.0)

# زر التوقع
if st.button("Predict"):

    payload = {
        "Payment_Method": payment_method,
        "Monthly_Charges": monthly_charges,
        "Total_Charges": str(total_charges)
    }

    try:
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)

        if response.status_code == 200:
            result = response.json()["prediction"]

            if result == 1:
                st.error("⚠️ Customer will CHURN")
            else:
                st.success("✅ Customer will NOT churn")

        else:
            st.error(f"API Error: {response.text}")

    except Exception as e:
        st.error(f"Connection Error: {e}") 