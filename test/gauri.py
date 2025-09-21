import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding

loan_df = pd.read_csv('databases/LC_loans_granting_model_dataset.csv')
sentiment_df = pd.read_csv('databases/sentiment_analysis_gasp.csv', encoding='ISO-8859-1')
lending_df = pd.read_csv('databases/train_lending_club.csv')

# CLEANING DATASET

print(loan_df.head())
print(sentiment_df.head())
print(lending_df.head())

print(loan_df.info())
print(sentiment_df.info())
print(lending_df.info())
df = loan_df.merge(sentiment_df, on="loan_id") \
            .merge(lending_df, on="loan_id")



X = df.drop(columns=['loan_status'])   # Features
y = df['loan_status']                  # Target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # sigmoid for binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc:.2f}")
predictions = model.predict(X_test[:5])
print(predictions)
predictions = model.predict(X_test[:5])
print(predictions)
 


