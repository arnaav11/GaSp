import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import sys
import xgboost as xgb


FILE_PATH = '/Users/sanidhya/Downloads/train_lending_club.csv'

try:
    df = pd.read_csv(FILE_PATH)
    print("Data loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{FILE_PATH}' was not found. Please check the file path.")
    sys.exit()


features_to_drop = ['id', 'issue_d', 'sub_grade']
df = df.drop(columns=features_to_drop)

 (the inputs)
target = 'loan_status'
y = df[target]
X = df.drop(columns=target)


numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns


numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])



neg_count = sum(y == 0)
pos_count = sum(y == 1)

scale_pos_weight_value = neg_count / pos_count


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', xgb.XGBClassifier(
                            n_estimators=100,
                            random_state=42,
                            n_jobs=-1,
                            scale_pos_weight=scale_pos_weight_value # Fix for imbalanced data
                        ))])


print("\nTraining the improved XGBoost model...")
model.fit(X_train, y_train)
print("Model training complete.")


y_pred = model.predict(X_test)

print("\n--- Improved XGBoost Model Evaluation ---")
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))


model_file = 'loan_status_model_fixed.joblib'
joblib.dump(model, model_file)
print(f"\nImproved XGBoost model saved to '{model_file}'.")