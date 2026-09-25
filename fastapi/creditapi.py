# uvicorn creditapi:app --reload

import os
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = os.path.join("..", "1B-Pipeline", "best_model", "model.pkl")
try:
    print(f"[API] Loading local model pickle from: {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("[API] Model loaded successfully!")
except Exception as e:
    print(f"[ERROR] Loading model failed: {e}")
    model = None

TARGET_CLASSES = ["Good", "Poor", "Standard"]
app = FastAPI(title="Credit Score Prediction API", version="1.0")

class CreditFeatures(BaseModel):
    Credit_Mix: str
    Payment_of_Min_Amount: str
    Occupation: str
    Payment_Behaviour: str
    Age: int
    Annual_Income: float
    Monthly_Inhand_Salary: float
    Num_Bank_Accounts: int
    Num_Credit_Card: int
    Interest_Rate: float
    Num_of_Loan: int
    Delay_from_due_date: int
    Num_of_Delayed_Payment: int
    Changed_Credit_Limit: float
    Num_Credit_Inquiries: int
    Outstanding_Debt: float
    Credit_Utilization_Ratio: float
    Total_EMI_per_month: float
    Amount_invested_monthly: float
    Monthly_Balance: float
    Credit_History_Age_Months: int


@app.get("/")
def root():
    return {
        "message": "Welcome to the Credit Score Prediction",
        "status": "Ready" if model is not None else "Model Not Loaded"
    }


@app.post("/predict")
def predict(features: CreditFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model pipeline is not loaded")
    
    try:
        data_in = features.dict()
        input_df = pd.DataFrame([data_in])
        
        ord_cols = ['Credit_Mix', 'Payment_of_Min_Amount']
        nom_cols = ['Occupation', 'Payment_Behaviour']
        num_cols = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
            'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months'
        ]
        
        input_df = input_df[ord_cols + nom_cols + num_cols]
        
        input_df['Interest_Rate'] = input_df['Interest_Rate'].astype('int64')
        input_df['Num_Credit_Inquiries'] = input_df['Num_Credit_Inquiries'].astype('float64')
        
        pred_encoded = model.predict(input_df)[0]
        pred_label = TARGET_CLASSES[int(pred_encoded)]
        
        return {
            "status": "success",
            "prediction_encoded": int(pred_encoded),
            "prediction": pred_label
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")