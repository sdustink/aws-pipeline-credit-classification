import joblib
import numpy as np
import pandas as pd
import json
import os
import io

COLUMNS = [
    "Credit_Mix", "Payment_of_Min_Amount",          # ord_cols
    "Occupation", "Payment_Behaviour",               # nom_cols
    "Age", "Annual_Income", "Monthly_Inhand_Salary",
    "Num_Bank_Accounts", "Num_Credit_Card", "Interest_Rate",
    "Num_of_Loan", "Delay_from_due_date", "Num_of_Delayed_Payment",
    "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
    "Credit_Utilization_Ratio", "Total_EMI_per_month",
    "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Age_Months",
]
NUM_COLS = COLUMNS[4:]
LABEL_MAP = {0: "Good", 1: "Poor", 2: "Standard"}


def model_fn(model_dir):
    return joblib.load(os.path.join(model_dir, "model.joblib"))


def input_fn(request_body, content_type):
    if content_type == "text/csv":
        df = pd.read_csv(io.StringIO(request_body), header=None, names=COLUMNS)
        df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        return df

    if content_type == "application/json":
        payload = json.loads(request_body)

        # support {"instances": [ {...}, {...} ]} format
        instances = payload.get("instances", [payload])
        df = pd.DataFrame(instances, columns=COLUMNS)
        df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").fillna(0)
        return df

    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    return model.predict(input_data)


def output_fn(prediction, accept):
    labels = [LABEL_MAP.get(int(p), str(p)) for p in prediction]
    result = {"predictions": [{"prediction": label} for label in labels]}
    return json.dumps(result), "application/json"