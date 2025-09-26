# =========================
# FULL CLIENT ANALYSIS SCRIPT
# =========================
import pandas as pd
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# -------------------------
# 1. Load datasets
# -------------------------
f_debt = pd.read_csv('test/databases/all_debt_reports2.csv')
df_profiles = pd.read_csv('test/databases/all_profiles2.csv')
df_bank = pd.read_csv('test/databases/all_statements2.csv')

# Merge datasets on client_id
df_clients = pd.merge(f_debt, df_profiles, on='client_id', how='outer')
df_clients = pd.merge(df_clients, df_bank, on='client_id', how='left')

# Drop rows with critical missing values
df_clients.dropna(subset=['annual_income','loan_amount_requested',
                          'collateral_value','current_balance'], inplace=True)

# -------------------------
# 2. Fill numeric fields
# -------------------------
features = [
    'credit_score', 'annual_income', 'loan_amount_requested',
    'collateral_value', 'alimony_payments_monthly',
    'current_balance', 'monthly_payment', 'is_delinquent'
]

for col in features:
    if col not in df_clients.columns:
        df_clients[col] = 0
    df_clients[col] = df_clients[col].fillna(0)

df_clients['is_delinquent'] = df_clients['is_delinquent'].astype(int)

# Derived ratios
df_clients['loan_to_income'] = df_clients['loan_amount_requested'] / (df_clients['annual_income'] + 1)
df_clients['balance_to_collateral'] = df_clients['current_balance'] / (df_clients['collateral_value'] + 1)

# -------------------------
# 3. Load pre-trained sentiment model
# -------------------------
sentiment_model_path = r'test/models/sentiment_analysis.keras'
tokenizer_path = r"test/models/sentiment_analysis_tokenizer.pickle"
max_length = 40

sentiment_model = tf.keras.models.load_model(sentiment_model_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

def get_sentiment_score(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.5
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_length, padding='post', truncating='post')
    score = sentiment_model.predict(padded, verbose=0)[0][0]
    return float(score)

# -------------------------
# 4. Load trained client score model + scaler
# -------------------------
client_model = tf.keras.models.load_model("test/models/sentiment_analysis.keras")
with open("test/models/sentiment_analysis_tokenizer.pickle", "rb") as handle:
    scaler = pickle.load(handle)

# -------------------------
# 5. Compute sentiment score for all clients
# -------------------------
if 'description' in df_clients.columns:
    df_clients['sentiment_score'] = df_clients['description'].apply(get_sentiment_score)
else:
    df_clients['sentiment_score'] = 0.5

# -------------------------
# 6. Predict client score
# -------------------------
model_features = features + ['loan_to_income','balance_to_collateral','sentiment_score']
X = df_clients[model_features].values
X_scaled = scaler.transform(X)

df_clients['predicted_client_score'] = client_model.predict(X_scaled, verbose=0).flatten()

# -------------------------
# 7. Human-readable assessment
# -------------------------
def get_assessment(score):
    return "Low risk client." if score > 0.5 else "Higher risk client."

df_clients['assessment'] = df_clients['predicted_client_score'].apply(get_assessment)

# -------------------------
# 8. Example: print first 5 clients
# -------------------------
for _, row in df_clients.head(5).iterrows():
    print(f"Client ID: {row['client_id']}")
    print(f"Sentiment Score: {row['sentiment_score']:.2f}")
    print(f"Predicted Client Score: {row['predicted_client_score']:.2f}")
    print(f"Assessment: {row['assessment']}\n")
