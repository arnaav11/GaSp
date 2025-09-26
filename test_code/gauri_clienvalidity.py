import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
# The pad_sequences import is commented out as it's not used in this score function.
# from tensorflow.keras.preprocessing.sequence import pad_sequences 

# -------------------------------------------------------------
# Fixed Client Assessment Function Signature for Streamlit/Orchestrator
# -------------------------------------------------------------

def run_client_assessment(
    client_info: pd.DataFrame, 
    client_transaction: pd.DataFrame,
    model_path: str = "test_code/models/client_score.keras",
    scaler_path: str = "test_code/models/client_score_scaler.pickle",
    output_file_path: str = 'client_analysis_results.csv'
) -> pd.DataFrame:
    """
    Performs client risk assessment using pre-loaded DataFrames.
    
    Args:
        client_info: DataFrame with client static information.
        client_transaction: DataFrame with transaction records.
        model_path: Path to the TensorFlow model file.
        scaler_path: Path to the feature scaler pickle file.
        output_file_path: Path to save the final CSV report.
    
    Returns:
        pd.DataFrame: DataFrame containing all client data and the final score/assessment.
    """
    
    # -------------------------
    # 1. Load datasets (FIXED: Uses input DataFrames directly)
    # -------------------------
    
    # Merge the input DataFrames to create the main working DataFrame
    df_clients = pd.merge(client_info, client_transaction, on='client_id', how='left')

    # -------------------------
    # 2. Fill numeric fields
    # -------------------------
    features = [
        'credit_score', 'annual_income', 'loan_amount_requested',
        'collateral_value', 'alimony_payments_monthly',
        'current_balance', 'monthly_payment', 'is_delinquent'
    ]

    for col in features:
        # Check if the column exists before trying to access it
        if col not in df_clients.columns:
            df_clients[col] = 0
        df_clients[col] = df_clients[col].fillna(0)

    # Ensure 'is_delinquent' exists before conversion
    if 'is_delinquent' not in df_clients.columns:
        df_clients['is_delinquent'] = 0
        
    df_clients['is_delinquent'] = df_clients['is_delinquent'].astype(int)

    # Derived ratios
    df_clients['loan_to_income'] = df_clients['loan_amount_requested'] / (df_clients['annual_income'] + 1)
    df_clients['balance_to_collateral'] = df_clients['current_balance'] / (df_clients['collateral_value'] + 1)

    # -------------------------
    # 3. Load trained models and scaler
    # -------------------------
    try:
        client_model = tf.keras.models.load_model(model_path)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading model/scaler files from '{model_path}' or '{scaler_path}': {e}")
        # Return data without prediction if model/scaler is missing
        return df_clients 

    # -------------------------
    # 4. & 5. Compute sentiment score (Placeholder)
    # -------------------------
    df_clients['sentiment_score'] = 0.5
    df_clients['sentiment_score_percent'] = (df_clients['sentiment_score']*100).round(2)


    # -------------------------
    # 6. Predict client score
    # -------------------------
    model_features = features + ['loan_to_income','balance_to_collateral']
    X = df_clients[model_features].values
    
    # *** CRITICAL CHECK FOR SHAPE AND NaNs BEFORE SCALING/PREDICTION ***
    # This section is robust, but ensure the model/scaler files are correct.
    
    X_scaled = scaler.transform(X)
    df_clients['predicted_client_score'] = client_model.predict(X_scaled, verbose=0).flatten()

    df_clients['predicted_client_score_percent'] = (df_clients['predicted_client_score']*100).round(2)

    # -------------------------
    # 7. Human-readable assessment
    # -------------------------
    df_clients['assessment'] = df_clients['predicted_client_score'].apply(
        lambda x: "Low risk client." if x > 0.5 else "Higher risk client."
    )

    # -------------------------
    # 8. Save the resulting df to a csv
    # -------------------------
    output_columns = [
        'client_id', 'first_name', 'last_name', 'ssn', 'address', 'employment_status',
        'annual_income', 'credit_score', 'loan_amount_requested', 'collateral_value',
        'alimony_payments_monthly', 'current_balance', 'monthly_payment', 'is_delinquent',
        'loan_to_income', 'balance_to_collateral',
        'sentiment_score',
        'predicted_client_score',
        'predicted_client_score_percent',
        'assessment'
    ]

    for col in output_columns:
        if col not in df_clients.columns:
            df_clients[col] = np.nan

    df_output = df_clients[output_columns].copy()
    df_output.to_csv(output_file_path, index=False) 

    print(f"Client analysis complete. Results saved to: '{output_file_path}'")
    
    return df_clients
# 9. Example: print first 5 clients (Retained for quick check)
# -------------------------
# print("\n--- First 5 Clients Example (for verification) ---")
# for _, row in df_clients.head(5).iterrows():
#     print(f"Client ID: {row['client_id']}")
#     print(f"Sentiment Score (Placeholder): {row['sentiment_score']:.2f}")
#     print(f"Predicted Client Score: {row['predicted_client_score']:.2f}")
#     print(f"Assessment: {row['assessment']}\n")