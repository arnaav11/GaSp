import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import pickle

# Load your client score model (already trained) + scaler
client_model = tf.keras.models.load_model("test/models/client_score.keras")
with open("test/models/client_score_scaler.pickle", "rb") as handle:
    scaler = pickle.load(handle)

features = [
    'credit_score', 'annual_income', 'loan_amount_requested',
    'collateral_value', 'alimony_payments_monthly',
    'current_balance', 'monthly_payment', 'is_delinquent'
]

def analyze_new_client(new_client_dict):
    df_new = pd.DataFrame([new_client_dict])
    
    for col in features:
        if col not in df_new.columns:
            df_new[col] = 0
        df_new[col] = df_new[col].fillna(0)

    df_new['is_delinquent'] = df_new['is_delinquent'].astype(int)
    df_new['loan_to_income'] = df_new['loan_amount_requested'] / (df_new['annual_income'] + 1)
    df_new['balance_to_collateral'] = df_new['current_balance'] / (df_new['collateral_value'] + 1)
    
    if 'description' in df_new.columns:
        df_new['sentiment_score'] = df_new['description'].apply(get_sentiment_score)
    else:
        df_new['sentiment_score'] = 0.5

    model_features = features + ['loan_to_income','balance_to_collateral','sentiment_score']
    X_new = df_new[model_features].values
    X_new_scaled = scaler.transform(X_new)

    df_new['predicted_client_score'] = client_model.predict(X_new_scaled, verbose=0).flatten()

    # Human-readable assessment
    results = []
    for _, row in df_new.iterrows():
        s = f"Client ID: {row['client_id']}\n"
        s += f"Sentiment Score: {row['sentiment_score']:.2f}\n"
        s += f"Predicted Client Score: {row['predicted_client_score']:.2f}\n"
        s += "Assessment: Low risk client.\n" if row['predicted_client_score'] > 0.5 else "Assessment: Higher risk client.\n"
        results.append(s)
    return results

