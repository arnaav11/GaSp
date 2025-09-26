import sys
import os
import pandas as pd
import numpy as np
from gauri_clienvalidity import run_client_assessment

    # 2. Import for output merging/saving
from test_merge_arnav import generate_all_client_pdfs

from pdf_to_csv_debug import extract_loan_data_to_dfs
# -----------------------------------------------------------------
## 1. Directory Setup and Imports
# -----------------------------------------------------------------

# Define the base directory (where all your custom modules are located)
# BASE_MODULE_DIR = r'C:\Users\Gauri\GaSp' 
TEST_SUBDIR = 'test_code' 

# Add the base directory to the system path to find modules:
# if BASE_MODULE_DIR not in sys.path: 
#     sys.path.append(BASE_MODULE_DIR)
#     print(f"SETUP: Added base module directory to path: {BASE_MODULE_DIR}")
    
# --- Module Imports ---

print("SETUP: All custom modules imported successfully.")

# -----------------------------------------------------------------
## 2. Defined Pipeline Step Functions (Data Flow Updated)
# -----------------------------------------------------------------

def step_1_data_receiver(filepaths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Acts as the initial data handler/validator for DataFrames received from the UI.
    """
    print("\n[STEP 1/3] Data received and initialized.")
    # print(f"  -> Initialized {len(df_client_info)} client info records and {len(df_transactions)} transaction records.")
    print(f"  -> Processing file: {filepaths}")
    os.listdir()
    return extract_loan_data_to_dfs(filepaths)

def step_2_analyze(df_client_info: pd.DataFrame, df_transactions: pd.DataFrame) -> pd.DataFrame:
    """
    Performs the core client validity analysis using the imported 'run_client_assessment' 
    function, passing it both required DataFrames.
    """
    print("\n[STEP 2/3] Running client validity analysis...")
    print(f"  -> Analyzing {len(df_client_info)} clients with {len(df_transactions)} transactions...")
    print(df_client_info.head())
    print(df_transactions.head())
    df_results = run_client_assessment(df_client_info, df_transactions) 
    print("  -> Analysis complete.")
    return df_results

# FIX: Added df_transactions as a required argument
def step_3_save_output(df_results: pd.DataFrame, df_transactions: pd.DataFrame, output_path: str):
    """
    Saves the final analysis results using the imported create_client_report function.
    """
    print("\n[STEP 3/3] Merging and saving final report...")
    # FIX: Pass the raw transactions data, as required by create_client_report
    generate_all_client_pdfs(df_results, df_transactions, output_path) 
    print(f"  -> Final report successfully saved to: {output_path}")

# -----------------------------------------------------------------
## 3. The Streamlit Entry Point (Core Logic)
# -----------------------------------------------------------------

def run_gasp_pipeline(filepaths: list[str], output_path: str = 'output/') -> str:
    """
    The main callable function for your Streamlit application. 
    It runs the entire analysis using pre-loaded DataFrames.
    """
    print("\nThank you for choosing GA$P. We are processing your request...")
    
    try:
        # 1. Initialize Data
        df_info, df_trans = step_1_data_receiver(filepaths)
        
        # 2. Analyze Data
        analysis_results = step_2_analyze(df_info, df_trans)
        
        # 3. Generate Output
        # FIX: Pass df_trans to the output step
        step_3_save_output(analysis_results, df_trans, output_path)
        
    except Exception as e:
        print(f"\nFATAL ERROR encountered during pipeline execution: {e}")
        raise e
        
    print("\nGA$P process successfully completed.")
    return output_path

# -----------------------------------------------------------------
## 4. Local Execution Block (Runs with Dummy Data for Testing)
# -----------------------------------------------------------------

def main():
    """
    Function to run the pipeline locally using dummy data for debugging.
    """
    
    
    try:
        # Define I/O File Paths
        FINAL_OUTPUT_PATH = 'output/gasp_client_report.csv'
        
        # 1. Create Dummy Data 
        df_info_dummy = pd.DataFrame({
            'client_id': [1, 2, 3],
            'first_name': ['Anna', 'Ben', 'Chris'],
            'last_name': ['Smith', 'Jones', 'Doe'],
            'annual_income': [100000.0, 50000.0, 75000.0],
            'credit_score': [750, 600, 700],
            'ssn': ['123-45-6789', '987-65-4321', '555-55-5555'],
            'address': ['123 Main St', '456 Oak Ave', '789 Pine Ln'],
            'employment_status': ['Employed', 'Self-Employed', 'Contractor'],
            'loan_amount_requested': [10000.0, 5000.0, 7500.0],
            'collateral_value': [15000.0, 0.0, 10000.0],
            'alimony_payments_monthly': [0.0, 500.0, 0.0],
            # Note: run_client_assessment will add score/sentiment columns
        })
        df_trans_dummy = pd.DataFrame({
            'client_id': [1, 1, 2, 3, 3],
            'date': pd.to_datetime(['2024-09-01', '2024-09-15', '2024-09-20', '2024-09-05', '2024-09-10']),
            'description': ['Deposit', 'Rent', 'Grocery', 'Loan Pmt', 'Fee'],
            'type': ['CREDIT', 'DEBIT', 'DEBIT', 'DEBIT', 'DEBIT'],
            'amount': [5000.0, 1500.0, 150.0, 200.0, 10.0],
            'balance': [5000.0, 3500.0, 800.0, 1800.0, 1790.0],
            'current_balance': [3500.0, 3500.0, 800.0, 1790.0, 1790.0],
            'monthly_payment': [0.0, 0.0, 0.0, 200.0, 0.0],
            'is_delinquent': [0, 0, 1, 0, 0],
        })
        
        os.makedirs('output', exist_ok=True)
        
        # 2. Run the main pipeline with dummy data
        run_gasp_pipeline(df_info_dummy, df_trans_dummy, FINAL_OUTPUT_PATH)
    
    except Exception as e:
        print(f"Local execution failed (Check if test/models/ exists and column names are correct): {e}")

if __name__ == "__main__":
    main()