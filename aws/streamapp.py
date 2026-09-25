import streamlit as st
import pandas as pd
import boto3
import json

ENDPOINT_NAME = "credit-score-endpoint"
AWS_REGION = "us-east-1" 

@st.cache_resource
def get_sagemaker_client():
    return boto3.client("sagemaker-runtime", region_name=AWS_REGION)

runtime_client = get_sagemaker_client()

st.set_page_config(page_title="Credit Score App", layout="wide")

st.title("💳 Credit Score Classification App (AWS Version)")
st.markdown("---")


col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.subheader("📊 Finances")
    annual_income = st.number_input("Annual Income ($)", min_value=0.0, value=50000.0, step=1000.0)
    monthly_inhand_salary = st.number_input("Monthly Inhand Salary ($)", min_value=0.0, value=4000.0, step=100.0)
    total_emi_per_month = st.number_input("Total EMI per Month ($)", min_value=0.0, value=100.0, step=10.0)
    amount_invested_monthly = st.number_input("Amount Invested Monthly ($)", min_value=0.0, value=200.0, step=50.0)
    monthly_balance = st.number_input("Monthly Balance ($)", min_value=0.0, value=500.0, step=50.0)
    outstanding_debt = st.number_input("Outstanding Debt ($)", min_value=0.0, value=1000.0, step=100.0)

with col_mid:
    st.subheader("💳 Credit History")
    num_bank_accounts = st.slider("Num Bank Accounts", min_value=0, max_value=20, value=10)
    num_credit_card = st.slider("Num Credit Card", min_value=0, max_value=20, value=10)
    interest_rate = st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, value=10.0)
    num_of_loan = st.slider("Num of Loan", min_value=0, max_value=20, value=10)
    delay_from_due_date = st.number_input("Delay from Due Date (Days)", min_value=0, max_value=100, value=5)
    num_of_delayed_payment = st.slider("Num of Delayed Payment", min_value=0, max_value=50, value=25)
    changed_credit_limit = st.number_input("Changed Credit Limit (%)", min_value=0.0, max_value=100.0, value=5.0)
    num_credit_inquiries = st.slider("Num Credit Inquiries", min_value=0, max_value=20, value=10)

with col_right:
    st.subheader("👤 Profile")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    credit_utilization_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
    credit_history_age_months = st.slider("Credit History Age (Months)", min_value=0, max_value=600, value=24)
    occupation = st.selectbox(
        "Occupation",
        ["Scientist", "Teacher", "Engineer", "Manager", "Developer", "Doctor", "Lawyer", "Media_Manager", "Architect", "Other"]
    )
    payment_behaviour = st.selectbox(
        "Payment Behaviour",
        [
            "Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
            "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments"
        ]
    )
    credit_mix = st.radio("Credit Mix", ["Good", "Standard", "Bad"], index=1, horizontal=True)
    payment_of_min_amount = st.radio("Payment of Min Amount", ["Yes", "No", "NM"], index=0, horizontal=True)

st.markdown("---")

if st.button("🚀 Make Prediction", type="primary", use_container_width=True):

    input_dict = {
        "Credit_Mix": credit_mix,
        "Payment_of_Min_Amount": payment_of_min_amount,
        "Occupation": occupation,
        "Payment_Behaviour": payment_behaviour,
        "Age": int(age),
        "Annual_Income": float(annual_income),
        "Monthly_Inhand_Salary": float(monthly_inhand_salary),
        "Num_Bank_Accounts": int(num_bank_accounts),
        "Num_Credit_Card": int(num_credit_card),
        "Interest_Rate": int(interest_rate),
        "Num_of_Loan": int(num_of_loan),
        "Delay_from_due_date": int(delay_from_due_date),
        "Num_of_Delayed_Payment": int(num_of_delayed_payment),
        "Changed_Credit_Limit": float(changed_credit_limit),
        "Num_Credit_Inquiries": float(num_credit_inquiries),
        "Outstanding_Debt": float(outstanding_debt),
        "Credit_Utilization_Ratio": float(credit_utilization_ratio),
        "Total_EMI_per_month": float(total_emi_per_month),
        "Amount_invested_monthly": float(amount_invested_monthly),
        "Monthly_Balance": float(monthly_balance),
        "Credit_History_Age_Months": int(credit_history_age_months),
    }

    input_df = pd.DataFrame([input_dict])

    with st.spinner("Sending request to SageMaker Endpoint..."):
        try:
            payload = input_df.to_csv(header=False, index=False).strip()

            response = runtime_client.invoke_endpoint(
                EndpointName=ENDPOINT_NAME,
                ContentType="text/csv",
                Body=payload
            )

            response_body = response["Body"].read().decode("utf-8")
            predictions = json.loads(response_body)

            pred_label = predictions[0] if isinstance(predictions, list) else predictions


            st.subheader("Prediction Result:")
            if pred_label in ["Good", 1, "1"]:
                st.success(f"🎯 Credit Score: **Good**")
            elif pred_label in ["Standard", 0, "0"]:
                st.info(f"⚠️ Credit Score: **Standard**")
            else:
                st.error(f"🚨 Credit Score: **Poor**")

        except Exception as e:
