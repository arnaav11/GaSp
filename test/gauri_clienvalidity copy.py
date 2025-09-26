import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.metrics import mean_squared_error, r2_score

# =====================
# Load Training Data
# =====================
df1 = pd.read_csv('test/databases/all_debt_reports2.csv')
df2 = pd.read_csv('test/databases/all_profiles2.csv')

df = pd.merge(df1, df2, on='client_id')

to_train = [
    'credit_score', 
    'annual_income',
    'loan_amount_requested', 
    'collateral_value',
    'alimony_payments_monthly',
    'current_balance',
    'monthly_payment',
    'is_delinquent'
]

to_predict = ['sentiment_score']

X = df[to_train].values
y = df[to_predict].values   # regression target

# Scale inputs
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Split dataset: train + "new data" for testing predictions
X_train, X_new, y_train, y_new_true = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# =====================
# Build and Train Model
# =====================
def build_regression_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1)  # predict sentiment_score
    ])
    return model

model = build_regression_model((X_train.shape[1],))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

# history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=64)

# Save model and scaler
model.save("sentiment_regressor.keras")
np.save("scaler_mean.npy", scaler_X.mean_)
np.save("scaler_scale.npy", scaler_X.scale_)

# =====================
# Predict on "New Data" from dataset
# =====================
loaded_model = tf.keras.models.load_model("sentiment_regressor.keras")

y_new_pred = loaded_model.predict(X_new)

# Create a DataFrame to compare predictions vs true sentiment
df_test = df.iloc[X_new.shape[0]*-1:].copy()  # last rows used as "new data"
df_test['predicted_sentiment'] = y_new_pred.flatten()

print(df_test[['client_id', 'predicted_sentiment', 'sentiment_score']].head(10))

# Optional: metrics
print("MSE on new data:", mean_squared_error(y_new_true, y_new_pred))
print("R² on new data:", r2_score(y_new_true, y_new_pred))
