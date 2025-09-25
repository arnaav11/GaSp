# =========================
# CLIENT ANALYSIS PIPELINE
# =========================
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Load datasets
# -------------------------
f_debt = pd.read_csv('test/databases/all_debt_reports2.csv')
df_profiles = pd.read_csv('test/databases/all_profiles2.csv')
df_bank = pd.read_csv('test/databases/all_statements2.csv')

# Merge debt + profile on client_id
df_clients = pd.merge(f_debt, df_profiles, on='client_id', how='outer')
# Merge bank statements (if needed)
df_clients = pd.merge(df_clients, df_bank, on='client_id', how='left')

# Drop rows with critical missing values
df_clients.dropna(subset=['annual_income','loan_amount_requested','collateral_value','current_balance'], inplace=True)

# -------------------------
# 2. Load trained sentiment model
# -------------------------
sentiment_model = tf.keras.models.load_model('test/models/sentiment_analysis.keras')
with open('test/models/sentiment_analysis_tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

vocab_size = 20000
max_length = 40

def get_sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    score = sentiment_model.predict(padded, verbose=0)[0][0]
    return float(score)  # 0-1 probability

# -------------------------
# 3. Prepare features
# -------------------------
features = [
    'credit_score', 'annual_income', 'loan_amount_requested',
    'collateral_value', 'alimony_payments_monthly',
    'current_balance', 'monthly_payment', 'is_delinquent'
]

# Fill missing numeric fields
for col in features:
    if col not in df_clients.columns:
        df_clients[col] = 0
    df_clients[col] = df_clients[col].fillna(0)

# Ensure is_delinquent is int
df_clients['is_delinquent'] = df_clients['is_delinquent'].astype(int)

# Add derived ratios
df_clients['loan_to_income'] = df_clients['loan_amount_requested'] / (df_clients['annual_income'] + 1)
df_clients['balance_to_collateral'] = df_clients['current_balance'] / (df_clients['collateral_value'] + 1)

# -------------------------
# 4. Train client score model
# -------------------------
# Use sentiment as target here if you want combined assessment, or create a numeric "client_score"
df_clients['client_score'] = df_clients['current_balance'] / (df_clients['annual_income'] + 1)  # example proxy

X = df_clients[features + ['loan_to_income','balance_to_collateral']].values
y = df_clients['client_score'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

client_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
client_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
client_model.fit(X_scaled, y, epochs=20, batch_size=64, verbose=1)

# -------------------------
# 5. Function to analyze new client
# -------------------------
def analyze_new_client(new_client_dict):
    df_new = pd.DataFrame([new_client_dict])
    
    # Fill missing numeric fields
    for col in features:
        if col not in df_new.columns:
            df_new[col] = 0
        df_new[col] = df_new[col].fillna(0)
    
    df_new['is_delinquent'] = df_new['is_delinquent'].astype(int)
    df_new['loan_to_income'] = df_new['loan_amount_requested'] / (df_new['annual_income'] + 1)
    df_new['balance_to_collateral'] = df_new['current_balance'] / (df_new['collateral_value'] + 1)
    
    # Sentiment score from description
    if 'description' in df_new.columns:
        df_new['sentiment_score'] = df_new['description'].apply(get_sentiment_score)
    else:
        df_new['sentiment_score'] = 0.5
    
    # Predict client score
    X_new = df_new[features + ['loan_to_income','balance_to_collateral']].values
    X_new_scaled = scaler.transform(X_new)
    df_new['predicted_client_score'] = client_model.predict(X_new_scaled).flatten()
    
    # Human-readable assessment
    assessment = []
    for _, row in df_new.iterrows():
        s = f"Client ID: {row['client_id']}\n"
        s += f"Sentiment Score (from statements): {row['sentiment_score']:.2f}\n"
        s += f"Predicted Client Score: {row['predicted_client_score']:.2f}\n"
        if row['predicted_client_score'] > 0.5:
            s += "Assessment: Low risk client.\n"
        else:
            s += "Assessment: Higher risk client.\n"
        assessment.append(s)
    return assessment

# -------------------------
# 6. Example usage
# -------------------------
new_client_example = {
    'client_id': 99999,
    'credit_score': 720,
    'annual_income': 85000,
    'loan_amount_requested': 15000,
    'collateral_value': 20000,
    'alimony_payments_monthly': 500,
    'current_balance': 10000,
    'monthly_payment': 400,
    'is_delinquent': 0,
    'description': 'Payroll direct deposit, good account activity.'
}

results = analyze_new_client(new_client_example)
for r in results:
    print(r)
