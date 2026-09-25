import shutil
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, RobustScaler

warnings.filterwarnings("ignore")

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = Path(filepath)

    def load(self) -> pd.DataFrame:
        print("\n*Step 1: Data Loading*")
        if not self.filepath.exists():
            raise FileNotFoundError(f"ERROR: '{self.filepath}' not found")
        print(f"'{self.filepath}' Found")
        df = pd.read_csv(self.filepath)
        print(df.head(5), df.shape)
        print("*Step 1 SUCCESS*")
        return df

class DataPreprocessor:
    ORD_COLS = ["Credit_Mix", "Payment_of_Min_Amount"]
    NOM_COLS = ["Occupation", "Payment_Behaviour"]
    NUM_COLS = [
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Delay_from_due_date",
        "Num_of_Delayed_Payment", "Changed_Credit_Limit", "Num_Credit_Inquiries",
        "Outstanding_Debt", "Credit_Utilization_Ratio", "Total_EMI_per_month",
        "Amount_invested_monthly", "Monthly_Balance", "Credit_History_Age_Months",
    ]

    def __init__(self, df: pd.DataFrame):
        self.df = df 
        self.target_encoder = LabelEncoder()

    def clean(self):
        print("\n*Step2: Data Cleaning*")
        df = self.df.copy()

        print("Fixing Mismatch Dtypes...")
        df["Age"] = df["Age"].fillna("0").str.extract(r"(\d+)").astype(float).astype(int)
        df["Num_of_Loan"] = df["Num_of_Loan"].fillna("0").str.extract(r"(\d+)").astype(float).astype(int)
        df["Num_of_Delayed_Payment"] = (
            df["Num_of_Delayed_Payment"].fillna("0").str.extract(r"(\d+)").astype(float).astype(int)
        )
        df["Annual_Income"] = df["Annual_Income"].str.replace(r"[^0-9.]", "", regex=True)
        df["Annual_Income"] = df["Annual_Income"].astype(float)
        
        df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].replace("_", np.nan)
        df["Changed_Credit_Limit"] = pd.to_numeric(df["Changed_Credit_Limit"], errors="coerce")
        df["Changed_Credit_Limit"] = df["Changed_Credit_Limit"].fillna(0)
        
        df["Outstanding_Debt"] = df["Outstanding_Debt"].astype(str)
        df["Outstanding_Debt"] = df["Outstanding_Debt"].str.replace(r"[^0-9.]", "", regex=True)
        df["Outstanding_Debt"] = pd.to_numeric(df["Outstanding_Debt"], errors="coerce")
        df["Outstanding_Debt"] = df["Outstanding_Debt"].fillna(0)
        
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].astype(str)
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].replace("", "0")
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].str.replace(r"[^0-9.]", "")
        df["Amount_invested_monthly"] = pd.to_numeric(df["Amount_invested_monthly"], errors="coerce")
        df["Amount_invested_monthly"] = df["Amount_invested_monthly"].fillna(0)
        
        df["Monthly_Balance"] = df["Monthly_Balance"].astype(str)
        df["Monthly_Balance"] = df["Monthly_Balance"].str.replace(r"[^0-9.-]+", "")
        df["Monthly_Balance"] = pd.to_numeric(df["Monthly_Balance"], errors="coerce")
        df["Monthly_Balance"] = df["Monthly_Balance"].fillna(0)
        
        extracted = df["Credit_History_Age"].str.extract(r"(?P<years>\d+)\s*Years\s*and\s*(?P<months>\d+)\s*Months")
        extracted = extracted.fillna(0).astype(int)
        df["Credit_History_Age_Months"] = (extracted["years"] * 12) + extracted["months"]
        df.drop(columns=["Credit_History_Age"], inplace=True)

        print("Dropping Duplicates...")
        df.drop_duplicates(inplace=True)

        print("Dropping Extreme Values...")
        int_cols = df.select_dtypes(include=["int64", "int32"]).columns.tolist()
        for column in int_cols:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        print("Dropping Illogical Values...")
        df = df[df["Age"] < 90]
        df = df[df["Num_Credit_Card"] <= 15]
        df = df[df["Interest_Rate"] <= 50]
        df = df[df["Num_of_Loan"] <= 15]
        df = df[(df["Num_Bank_Accounts"] > 0) & (df["Num_Bank_Accounts"] <= 10)]
        df = df[(df["Delay_from_due_date"] > 0) & (df["Delay_from_due_date"] <= 60)]
        df = df[(df["Changed_Credit_Limit"] > 0) & (df["Changed_Credit_Limit"] <= 30)]
        df = df[df["Num_Credit_Inquiries"] <= 15]
        df = df[df["Total_EMI_per_month"] <= 200]
        df = df[df["Outstanding_Debt"] <= 1500]

        df.drop(columns=["Unnamed: 0"], inplace=True)
        df = df[df["Payment_Behaviour"] != "!@9#%8"]
        df = df[df["Occupation"] != "_______"]
        df = df[df["Credit_Mix"] != "_"]
        df.dropna(inplace=True)

        print("Dropping Useless Columns...")
        df.drop(columns=["ID", "Customer_ID", "Month", "Name", "SSN", "Type_of_Loan"], inplace=True)

        print(df.head())
        print("*Step2 SUCCESS*")

        self.df = df
        return self

    def split(self, test_size: float = 0.2, random_state: int = 42):
        print("\n*Step3: Data Preprocessing*")
        print("Splitting Train-Test Data...")

        X = self.df[self.ORD_COLS + self.NOM_COLS + self.NUM_COLS]
        y = self.df["Credit_Score"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        print(f"X_train: {X_train.shape}, X_test: {X_test.shape}\n")

        print("Applying RandomOverSampler to Training Data...")
        from imblearn.over_sampling import RandomOverSampler
        sampler = RandomOverSampler(random_state=random_state)
        X_train_resampled, y_train_resampled = sampler.fit_resample(X_train, y_train)
        
        print(f"Resampled X_train: {X_train_resampled.shape}")

        y_train_enc = self.target_encoder.fit_transform(y_train_resampled)
        y_test_enc = self.target_encoder.transform(y_test)

        print(f"X_train: {X_train_resampled.shape}, X_test: {X_test.shape}\n")
        return X_train_resampled, X_test, y_train_enc, y_test_enc

    def build_transformer(self) -> ColumnTransformer:
        print("Building ColumnTransformer...")
        return ColumnTransformer(
            transformers=[
                ("ord", OrdinalEncoder(), self.ORD_COLS),
                ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), self.NOM_COLS),
                ("num", RobustScaler(), self.NUM_COLS),
            ]
        )


class ModelTrainer:
    def __init__(self, preprocessor: ColumnTransformer, classifier, experiment_name: str, tracking_dir: Path):
        self.pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])
        mlflow.set_tracking_uri(tracking_dir.as_uri())
        mlflow.set_experiment(experiment_name)
        mlflow.sklearn.autolog()
        self.run = None

    def train(self, X_train, y_train) -> Pipeline:
        print("\n*Step4: Data Modeling*")
        print(self.pipeline)
        self.run = mlflow.start_run()
        self.pipeline.fit(X_train, y_train)
        print("*Step4 SUCCESS*")
        return self.pipeline

    def end_run(self):
        mlflow.end_run()


class ModelEvaluator:
    def __init__(self, pipeline: Pipeline, target_encoder: LabelEncoder):
        self.pipeline = pipeline
        self.target_encoder = target_encoder

    def evaluate(self, X_test, y_test, best_model_dir: str = "best_model") -> float:
        print("\n*Step5: Model Evaluation*")
        y_pred = self.pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("RFC Accuracy:", accuracy)

        eval_df = X_test.copy()
        eval_df["target_label"] = y_test
        test_dataset = mlflow.data.from_pandas(eval_df, targets="target_label", name="credit_score_test")

        print("Logging test metrics...")
        mlflow.evaluate(
            model=lambda data: self.pipeline.predict(data),
            data=test_dataset,
            model_type="classifier",
        )

        self._log_confusion_matrix(y_test, y_pred)

        report = classification_report(y_test, y_pred, target_names=self.target_encoder.classes_)
        print("\nClassification Report\n", report)

        self._save_best_model(best_model_dir)
        print("*Step5 SUCCESS*")
        return accuracy

    def _log_confusion_matrix(self, y_test, y_pred):
        matrix = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(
            matrix, annot=True, linewidth=0.5, fmt="d",
            xticklabels=self.target_encoder.classes_,
            yticklabels=self.target_encoder.classes_,
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("RFC Confusion Matrix")
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)
        plt.close()

    def _save_best_model(self, best_model_dir: str):
        path = Path(best_model_dir)
        if path.exists():
            shutil.rmtree(path)
        mlflow.sklearn.save_model(self.pipeline, str(path))
        mlflow.log_artifacts(str(path), artifact_path="best_model")
        print(f"Model terbaik disimpan di '{path}' dan dicatat sebagai artifact MLflow.")


def main():
    print("=== Pipeline Running ===")

    base_dir = Path(__file__).resolve().parent
    tracking_dir = base_dir / "mlruns"

    loader = DataLoader("data_A.csv")
    try:
        df_raw = loader.load()
    except FileNotFoundError as e:
        print(e)
        return

    prep = DataPreprocessor(df_raw)
    prep.clean()

    X_train, X_test, y_train, y_test = prep.split()
    transformer = prep.build_transformer()

    trainer = ModelTrainer(
        preprocessor=transformer,
        classifier=RandomForestClassifier(random_state=42, class_weight='balanced'),
        experiment_name="Credit_Score_Experiment",
        tracking_dir=tracking_dir,
    )
    pipeline = trainer.train(X_train, y_train)

    evaluator = ModelEvaluator(pipeline, prep.target_encoder)
    evaluator.evaluate(X_test, y_test)

    trainer.end_run()

if __name__ == "__main__":
    main()