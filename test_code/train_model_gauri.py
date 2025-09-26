

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# --------------------------
# 1. Load datasets
# --------------------------
df_debt = pd.read_csv('test/databases/all_debt_reports2.csv')
df_profiles = pd.read_csv('test/databases/all_profiles2.csv')
df_bank = pd.read_csv('test/databases/all_statements2.csv')

# Merge on client_id
df_clients = pd.merge(df_debt, df_profiles, on='client_id', how='outer')
df_clients = pd.merge(df_clients, df_bank, on='client_id', how='outer')

# Drop rows with critical missing values
df_clients.dropna(subset=['annual_income','loan_amount_requested','collateral_value','current_balance'], inplace=True)

# --------------------------
# 2. Fill numeric fields
# --------------------------
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

# --------------------------
# 3. Derived features
# --------------------------
df_clients['loan_to_income'] = df_clients['loan_amount_requested'] / (df_clients['annual_income'] + 1)
df_clients['balance_to_collateral'] = df_clients['current_balance'] / (df_clients['collateral_value'] + 1)

# --------------------------
# 4. Define client score (target)
# --------------------------
# Simple example: lower balance-to-income + lower delinquency -> better score
df_clients['client_score'] = 1 - (
    0.5*df_clients['loan_to_income'] + 
    0.3*df_clients['balance_to_collateral'] + 
    0.2*df_clients['is_delinquent']
)
# Clip between 0 and 1
df_clients['client_score'] = df_clients['client_score'].clip(0,1)

y = df_clients['client_score'].values

# --------------------------
# 5. Prepare features
# --------------------------
model_features = features + ['loan_to_income','balance_to_collateral']
X = df_clients[model_features].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --------------------------
# 6. Train-test split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.1, random_state=42)

# --------------------------
# 7. Build neural network
# --------------------------
client_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_scaled.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # output 0-1
])

client_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
client_model.summary()

# --------------------------
# 8. Train model
# --------------------------
history = client_model.fit(X_train, y_train, epochs=10, batch_size=45,validation_split=0.1, verbose=1)

# --------------------------
# 9. Evaluate model
# --------------------------
y_pred = client_model.predict(X_test).flatten()
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}, Test MAE: {mae:.4f}")

# --------------------------
# 10. Predict client scores for all clients
# --------------------------
df_clients['predicted_client_score'] = client_model.predict(X_scaled).flatten()

# Convert to percentage
df_clients['predicted_client_score_percent'] = (df_clients['predicted_client_score']*100).round(2)

# --------------------------
# 11. Sample output
# --------------------------
print(df_clients[['client_id','predicted_client_score','predicted_client_score_percent']].head(10))

# --------------------------
# 12. Save model and scaler
# --------------------------
client_model.save('test/models/client_score_model.keras')
with open('test/models/client_score_scaler.pickle','wb') as f:
    pickle.dump(scaler,f)
