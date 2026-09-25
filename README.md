# AWS Pipeline Credit Classification

An end-to-end **credit score classification project** covering data preprocessing, exploratory data analysis, machine learning model development, hyperparameter tuning, and deployment. The project includes both **cloud deployment using AWS SageMaker** and **local deployment using FastAPI and Streamlit**.

## 📊 Dataset

The project uses a credit-related dataset obtained from Kagggle: https://www.kaggle.com/datasets/parisrohan/credit-score-classification. It contains data including customer finances, credit history, and profile for classifying their credit scores into 3 categories (Good, Standard, Poor).

## ⚙️ Methodology

The project follows an end-to-end machine learning pipeline:

1. Perform **exploratory data analysis (EDA)** to understand the dataset and identify relevant patterns.
2. Clean and preprocess the data through an **ETL pipeline**.
3. Develop and evaluate machine learning classification models.
4. Perform **hyperparameter tuning** to improve model performance.
5. Build a reusable preprocessing and modeling pipeline.
6. Deploy the pipeline in 2 ways:

   1. **Local deployment**: deploy the trained model locally using **FastAPI** as the backend and **Streamlit** as the frontend.
   2. **Cloud deployment**: deploy the model to **AWS SageMaker** for cloud-based inference, connecting to a **Streamlit** frontend using an endpoint.

## 📈 Results

The final classification model predicts credit score categories based on features divided into 3 main aspects: finances (i.e. annual income, outstanding debt, etc), credit history (i.e. num of bank accounts, loans, etc), profile (i.e. age, payment behaviour, etc).

The project demonstrates an end-to-end workflow from **raw data processing and model development to local and cloud deployment**, providing both a local API-based application and an AWS-hosted inference endpoint.

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* FastAPI
* Streamlit
* AWS SageMaker
* Jupyter Notebook
* Git / GitHub

## 📂 Project Structure

```text
aws-pipeline-credit-classification/
├── aws/
│   └── AWS SageMaker deployment files
├── eda/
│   └── Exploratory data analysis and initial modeling
├── fastapi/
│   └── FastAPI backend and Streamlit frontend
└── pipeline/
    └── ETL pipeline and hyperparameter tuning
```
