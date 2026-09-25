import json
import os
import joblib
import pandas as pd

def model_fn(model_dir):
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        request_data = json.loads(request_body)
        
        instances = request_data.get("instances", [])
        df = pd.DataFrame(instances)
        
        ord_cols = ['Credit_Mix', 'Payment_of_Min_Amount']
        nom_cols = ['Occupation', 'Payment_Behaviour']
        num_cols = [
            'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
            'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
            'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
            'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
            'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months'
        ]
        
        df = df[ord_cols + nom_cols + num_cols]
        return df
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    prediction = model.predict(input_data)
    return prediction

def output_fn(prediction, response_content_type):
    if response_content_type == "application/json":
        TARGET_CLASSES = ["Good", "Poor", "Standard"]
        
        result = []
        for pred in prediction:
            result.append({
                "prediction_encoded": int(pred),
                "prediction": TARGET_CLASSES[int(pred)]
            })
            
        return json.dumps({"predictions": result}), response_content_type
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")