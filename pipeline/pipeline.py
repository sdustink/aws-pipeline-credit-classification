from pathlib import Path
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error

import warnings
warnings.filterwarnings("ignore")

def main():
    print("=== PIPELINE RUNNING ===")

    print("\n*Step1: Data Loading*")
    raw_file = Path("data_A.csv")
    if not raw_file.exists():
        print("ERROR: 'data_A.csv' not found")
        return
    else:
        print("'data_A.csv' Found")
    df = pd.read_csv(raw_file)
    print(df.head(5), df.shape)
    print("*Step1 SUCCESS*")


    print("\n*Step2: Data Cleaning*")
    print("Fixing Mismatch Dtypes...")
    df['Age'] = df['Age'].fillna('0').str.extract(r'(\d+)').astype(float).astype(int)
    df['Num_of_Loan'] = df['Num_of_Loan'].fillna('0').str.extract(r'(\d+)').astype(float).astype(int)
    df['Num_of_Delayed_Payment'] = df['Num_of_Delayed_Payment'].fillna('0').str.extract(r'(\d+)').astype(float).astype(int)
    df['Annual_Income'] = df['Annual_Income'].str.replace(r'[^0-9.]', '', regex=True)
    df['Annual_Income'] = df['Annual_Income'].astype(float)
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].replace('_', np.nan)
    df['Changed_Credit_Limit'] = pd.to_numeric(df['Changed_Credit_Limit'], errors='coerce')
    df['Changed_Credit_Limit'] = df['Changed_Credit_Limit'].fillna(0)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].astype(str)
    df['Outstanding_Debt'] = df['Outstanding_Debt'].str.replace(r'[^0-9.]', '', regex=True)
    df['Outstanding_Debt'] = pd.to_numeric(df['Outstanding_Debt'], errors='coerce')
    df['Outstanding_Debt'] = df['Outstanding_Debt'].fillna(0)
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].astype(str)
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].replace('', '0')
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].str.replace(r'[^0-9.]', '')
    df['Amount_invested_monthly'] = pd.to_numeric(df['Amount_invested_monthly'], errors='coerce')
    df['Amount_invested_monthly'] = df['Amount_invested_monthly'].fillna(0)
    df['Monthly_Balance'] = df['Monthly_Balance'].astype(str)
    df['Monthly_Balance'] = df['Monthly_Balance'].str.replace(r'[^0-9.-]+', '')
    df['Monthly_Balance'] = pd.to_numeric(df['Monthly_Balance'], errors='coerce')
    df['Monthly_Balance'] = df['Monthly_Balance'].fillna(0)
    extracted = df['Credit_History_Age'].str.extract(r'(?P<years>\d+)\s*Years\s*and\s*(?P<months>\d+)\s*Months')
    extracted = extracted.fillna(0).astype(int)
    df['Credit_History_Age_Months'] = (extracted['years'] * 12) + extracted['months']
    df.drop(columns=['Credit_History_Age'], inplace=True)

    print("Dropping Duplicates...")
    df.drop_duplicates(inplace=True)

    print("Dropping Extreme Values...")
    num_cols = df.select_dtypes(include=['int64', 'int32']).columns.tolist()
    for column in num_cols:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    print("Dropping Illogical Values...")
    df = df[df['Age'] < 90]
    df = df[df['Num_Credit_Card'] <= 15]
    df = df[df['Interest_Rate'] <= 50]
    df = df[df['Num_of_Loan'] <= 15]
    df = df[(df['Num_Bank_Accounts'] > 0) & (df['Num_Bank_Accounts'] <= 10)]
    df = df[(df['Delay_from_due_date'] > 0) & (df['Delay_from_due_date'] <= 60)]
    df = df[(df['Changed_Credit_Limit'] > 0) & (df['Changed_Credit_Limit'] <= 30)]
    df = df[df['Num_Credit_Inquiries'] <= 15]
    df = df[df['Total_EMI_per_month'] <= 200]
    df = df[df['Outstanding_Debt'] <= 1500]

    df.drop(columns=['Unnamed: 0'], inplace=True)
    df = df[df['Payment_Behaviour'] != '!@9#%8']
    df = df[df['Occupation'] != '_______']
    df = df[df['Credit_Mix'] != '_']
    df.dropna(inplace=True)

    print("Dropping Useless Columns...")
    df.drop(columns=['ID', 'Customer_ID', 'Month', 'Name', 'SSN', 'Type_of_Loan'], inplace=True)

    print(df.head())
    print("*Step2 SUCCESS*")


    print("\n*Step3: Data Preprocessing*")

    print("Splitting Train-Test Data...")
    ord_cols = ['Credit_Mix', 'Payment_of_Min_Amount']
    nom_cols = ['Occupation', 'Payment_Behaviour']
    num_cols = [
        'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 
        'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 
        'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 
        'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Total_EMI_per_month', 
        'Amount_invested_monthly', 'Monthly_Balance', 'Credit_History_Age_Months'
    ]

    X = df[ord_cols + nom_cols + num_cols]
    y = df['Credit_Score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")
    
    print("Making Pipeline...")
    target_encoder = LabelEncoder()
    y_train_clean = target_encoder.fit_transform(y_train)
    y_test_clean = target_encoder.transform(y_test)

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", OrdinalEncoder(), ord_cols),
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False),nom_cols),
            ("num", RobustScaler(), num_cols)
        ]
    )
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42))
        ]
    )
    print(pipeline)

    print("*Step3 SUCCESS*")

    print("\n*Step4: Data Modeling*")
    import mlflow

    base_dir = Path(__file__).resolve().parent
    tracking_uri = (base_dir / "mlruns").as_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Credit_Score_Experiment")
    mlflow.sklearn.autolog()

    with mlflow.start_run() as run:
        pipeline.fit(X_train, y_train_clean)

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test_clean, y_pred)
        print("RFC Accuracy:", accuracy)

        eval_df = X_test.copy()
        eval_df["target_label"] = y_test_clean

        test_dataset = mlflow.data.from_pandas(
            eval_df, targets="target_label", name="credit_score_test"
        )

        print("Logging test metrics...")

        mlflow.evaluate(
            model=lambda data: pipeline.predict(data),
            data=test_dataset,
            model_type="classifier",
        )

        matrix = confusion_matrix(y_test_clean, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            matrix,
            annot=True,
            linewidth=0.5,
            fmt="d",
            xticklabels=target_encoder.classes_,
            yticklabels=target_encoder.classes_,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("RFC Confusion Matrix")
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

        print("\nClassification Report\n", classification_report(y_test_clean, y_pred, target_names=target_encoder.classes_))
        
    print("*Step4 SUCCESS*")

if __name__ == "__main__":
    main()