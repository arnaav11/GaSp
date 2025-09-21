import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import os
import matplotlib

# --- 1. Data Ingestion & Preprocessing ---
try:
    # Load the dataset from the CSV file
    df = pd.read_csv('test\databases\LC_loans_granting_model_dataset.csv')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: The file 'LC_loans_granting_model_dataset.csv' was not found.")
    print("Please ensure the file is in the same directory as this script.")
    exit()

# Define the target variable and features
# You may need to change 'loan_status' if your column has a different name
target_variable_name = 'loan_status'
X = df.drop(target_variable_name, axis=1)
y = df[target_variable_name]

# Handle missing values for numerical columns
numerical_cols = X.select_dtypes(include=np.number).columns
imputer_numerical = SimpleImputer(strategy='mean')
X[numerical_cols] = imputer_numerical.fit_transform(X[numerical_cols])

# Handle categorical columns using one-hot encoding
categorical_cols = X.select_dtypes(include='object').columns
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData preprocessed and split successfully.")

# --- 2. Model Building and Training ---
# Initialize the XGBoost classifier
# This model is well-suited for structured, tabular data like a credit dataset
model = xgb.XGBClassifier(objective='binary:logistic',
                          n_estimators=100,
                          learning_rate=0.1,
                          use_label_encoder=False,
                          eval_metric='logloss',
                          random_state=42)

# Train the model on the training data
print("\nStarting model training with XGBoost...")
model.fit(X_train, y_train)

print("XGBoost model trained successfully.")

# --- 3. Model Evaluation ---
# Make predictions on the test set
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

# Calculate key performance metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\nModel Evaluation:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")