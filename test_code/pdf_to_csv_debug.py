import pdfplumber
import pandas as pd
import re
from io import StringIO
from typing import Optional, List, Dict, Any, Tuple # Added Tuple for type hint

# --- Helper Functions for Data Extraction and Cleaning ---

def clean_and_convert_currency(s: str) -> float:
    """Removes currency symbols and commas, then converts the string to a float."""
    if pd.isna(s):
        return 0.0
    # Use regex to remove common currency symbols and commas
    cleaned = re.sub(r'[$,]', '', str(s)).strip()
    try:
        return float(cleaned)
    except ValueError:
        return 0.0 # Return 0 if conversion fails

def calculate_derived_financial_data(df_transactions: pd.DataFrame) -> Dict[str, float]:
    """
    Analyzes the transaction history to derive financial metrics.
    
    Assumptions:
    1. Annual Income: Estimated by taking the largest 'Paycheck Deposit' or 
       'Bonus Payment' in the statement period and annualizing it (multiplying by 12 
       to represent monthly income).
    2. Alimony/Debt Payments: Assumed to be 0 unless a clear, consistent, and
       large recurring debit is identified as such, which is not the case here.
    """
    
    # 1. Standardize and clean the 'Amount' column
    df_transactions['Amount_Cleaned'] = df_transactions['Amount'].apply(clean_and_convert_currency)
    
    # 2. Estimate Annual Income
    income_keywords = ['Paycheck Deposit', 'Bonus Payment']
    income_transactions = df_transactions[
        (df_transactions['Type'] == 'CREDIT') & 
        (df_transactions['Description'].str.contains('|'.join(income_keywords), case=False, na=False))
    ]
    
    # Find the single largest income transaction as an estimate for monthly income
    if not income_transactions.empty:
        monthly_income_estimate = income_transactions['Amount_Cleaned'].max()
        annual_income_estimate = monthly_income_estimate * 12
    else:
        annual_income_estimate = 0.0
        
    # 3. Estimate Alimony Payments (default to 0, as not explicitly stated)
    alimony_payments_monthly = 0.0
    
    # 4. (Optional) Estimate Credit Score/Employment Status - Cannot be derived, use placeholders
    
    return {
        'annual_income': annual_income_estimate,
        'alimony_payments_monthly': alimony_payments_monthly
    }

# --- Main Conversion Function ---

def convert_bank_statement_to_loan_dataframes(pdf_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Converts a bank statement PDF into two pandas DataFrames:
    1. df_loan_data: Summary data matching the loan application schema.
    2. df_transactions: The raw transaction history extracted from the statement.

    Args:
        pdf_filepath: The path to the input PDF file (e.g., 'Bank_Statement_7_Johnson.pdf').
        
    Returns:
        A tuple containing (df_loan_data, df_transactions).
    """
    
    print(f"Processing PDF file: {pdf_filepath}")
    
    client_data: Dict[str, Any] = {
        'client_id': None, 
        'first_name': None, 
        'last_name': None
    }
    transaction_data_list: List[Dict[str, str]] = []

    # Initialize empty DataFrames for error handling/return signature consistency
    target_columns = [
        'client_id', 'first_name', 'last_name', 'ssn', 'address', 
        'annual_income', 'employment_status', 'credit_score', 
        'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly', 
        'sentiment_score'
    ]
    transaction_cols = ['Date', 'Description', 'Type', 'Amount', 'Balance'] # Estimated transaction columns
    
    df_loan_data_empty = pd.DataFrame(columns=target_columns)
    df_transactions_empty = pd.DataFrame(columns=transaction_cols)

    try:
        with pdfplumber.open(pdf_filepath) as pdf:
            # --- 1. Extract Client Data from the first page's text ---
            first_page_text = pdf.pages[0].extract_text()
            
            # Extract Client ID
            id_match = re.search(r'Client ID:\s*(\d+)', first_page_text)
            if id_match:
                client_data['client_id'] = int(id_match.group(1))

            # Extract Client Name (assuming format: **Client Name:** First Last)
            name_match = re.search(r'Client Name:\s*(\w+)\s*(\w+)', first_page_text)
            if name_match:
                client_data['first_name'] = name_match.group(1)
                client_data['last_name'] = name_match.group(2)

            # --- 2. Extract Transaction Table from all pages ---
            for page in pdf.pages:
                # The uploaded PDF snippet shows CSV-like text which is easy to parse
                # by simply extracting the raw text and using pandas read_csv on a StringIO buffer.
                raw_text = page.extract_text()
                
                # Find the start of the transaction table (after the header)
                # The raw text contains the header: "Date\n", "Description\n", ...
                table_start_match = re.search(r'"Date\s*"\s*,\s*"Description\s*"\s*,\s*"Type\s*"\s*,\s*"Amount\s*"\s*,\s*"Balance\s*"', raw_text, re.DOTALL)

                if table_start_match:
                    # Isolate the CSV data part
                    csv_data = raw_text[table_start_match.start():]
                    
                    # Clean up the raw text for proper CSV parsing
                    # Replace newline within quotes and strip
                    csv_io = StringIO(csv_data.replace('\n', '').replace('"', '').strip())
                    
                    # Read the CSV data directly. Skip the first row which is the header.
                    df_page = pd.read_csv(csv_io, header=0)
                    
                    # Clean the column names 
                    df_page.columns = transaction_cols

                    # Append transactions to the master list
                    transaction_data_list.append(df_page)

        # Concatenate all transaction data
        if not transaction_data_list:
             raise ValueError("Could not extract any transaction data from the PDF.")
             
        df_transactions = pd.concat(transaction_data_list, ignore_index=True)
        
        # --- 3. Calculate Derived Metrics ---
        derived_metrics = calculate_derived_financial_data(df_transactions)
        
        # --- 4. Assemble the Final Loan Application DataFrame ---
        
        # Create a dictionary for the single output row
        output_row = {
            'client_id': client_data.get('client_id'),
            'first_name': client_data.get('first_name'),
            'last_name': client_data.get('last_name'),
            
            # Extracted/Derived Financials
            'annual_income': round(derived_metrics['annual_income'], 2),
            'alimony_payments_monthly': round(derived_metrics['alimony_payments_monthly'], 2),
            
            # Placeholders for data not available in a bank statement
            'ssn': pd.NA,
            'address': pd.NA,
            'employment_status': 'Unknown (Bank Statement Only)',
            'credit_score': pd.NA,
            'loan_amount_requested': pd.NA,
            'collateral_value': pd.NA,
            'sentiment_score': pd.NA
        }
        
        df_loan_data = pd.DataFrame([output_row], columns=target_columns)
        
        # --- 5. Return DataFrames ---
        print("\nDataFrames successfully generated.")
        return df_loan_data, df_transactions
        
    except Exception as e:
        print(f"An error occurred during processing: {e}")
        # Return empty DataFrames on failure
        return df_loan_data_empty, df_transactions_empty

if __name__ == '__main__':
    # NOTE: In a real environment, you would call this function with the actual 
    # file path, like: convert_bank_statement_to_loan_dataframes("Bank_Statement_7_Johnson.pdf")
    
    # Simulate a call using the uploaded file path
    INPUT_PDF = "Bank_Statement_7_Johnson.pdf"
    
    # Execute the conversion
    loan_summary_df, transaction_history_df = convert_bank_statement_to_loan_dataframes(INPUT_PDF)

    print("\n--- Loan Applicant Summary DataFrame (df_loan_data) ---")
    print(loan_summary_df.to_markdown(index=False))
    
    print("\n--- Transaction History DataFrame (df_transactions) (First 5 Rows) ---")
    print(transaction_history_df.head().to_markdown(index=False))
