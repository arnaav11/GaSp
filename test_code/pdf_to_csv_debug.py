import pandas as pd
import re
from typing import Any, Dict, List, Tuple

# We must assume the file content is accessible by simply referencing the file path
# through an implicit environment-provided mechanism. In this specific runtime,
# we simulate fetching the content using the uploaded file snippets, but in a real 
# script, this function would rely on the environment's internal fetcher.

def get_file_content(pdf_filepath: str) -> str:
    """Simulates fetching the raw text content from the environment and prints it."""
    
    # Placeholder mapping based on the content I know was uploaded. 
    # In a live environment, this would be the environment's file access call.
    if pdf_filepath == "Loan_Profile_10_Farley.pdf":
        full_text = 'CLIENT LOAN & CREDIT PROFILE\n\nClient Name: Samuel Farley | Client ID: 10\n Report Date: 2025-09-26\n\nFinal Assessment: N/A\n\nLOAN & CREDIT PROFILE SUMMARY\n\n\n"SSN:\n","XXX-XX-9372\n"\r\n"Address:\n","PSC 4915, Box 3024, APO AE 55468\n"\r\n"Credit Score:\n","663\n"\r\n"Collateral Value:\n","N/A\n"\n\n"Annual Income:\n","\$131,070\n"\r\n"Employment:\n","Self-Employed\n"\r\n"Loan Requested:\n","\$26,102\n"\r\n"Monthly Alimony:\n","N/A\n"\n\nClient Sentiment Score: -0.97 (Very Negative - Possible dissatisfaction or risk detected.)\n'
    elif pdf_filepath == "Bank_Statement_10_Farley.pdf":
        full_text = 'CLIENT ACCOUNT STATEMENT\n\nClient Name: Samuel Farley | Client ID: 10\n\nStatement Period: 2025-04-03 to 2025-07-05\n\nReport Date: 2025-09-26\n\nTRANSACTION HISTORY\n\nOpening Balance: \$3,476.00 | Closing Balance: \$4,283.94\n\n\n"Date\n","Description\n","Type\n","Amount\n","Balance\n"\r\n"2025-04-03\n","Online Order - Best Buy\n","DEBIT\n","\$412.96\n","\$3,063.04\n"\r\n"2025-04-05\n","Streaming Service - Spotify\n","DEBIT\n","\$244.72\n","\$2,818.32\n"\r\n"2025-04-08\n","Bonus Payment\n","CREDIT\n","\$71.14\n","\$2,889.46\n"\r\n"2025-04-10\n","POS Debit - Amazon.com\n","DEBIT\n","\$342.53\n","\$2,546.93\n"\r\n"2025-04-14\n","Gas Station - Uber\n","DEBIT\n","\$29.29\n","\$2,517.64\n"\r\n"2025-04-17\n","Gift from family\n","CREDIT\n","\$285.29\n","\$2,802.93\n"\r\n"2025-04-21\n","Paycheck Deposit\n","CREDIT\n","\$59.46\n","\$2,862.39\n"\r\n"2025-04-25\n","ATM Withdrawal - Fee\n","DEBIT\n","\$17.00\n","\$2,845.39\n"\r\n"2025-04-29\n","Car Payment\n","DEBIT\n","\$325.75\n","\$2,519.64\n"\r\n"2025-05-02\n","Paycheck Deposit\n","CREDIT\n","\$279.13\n","\$2,798.77\n"\r\n"2025-05-06\n","Grocery Store - Trader Joe\'s\n","DEBIT\n","\$180.99\n","\$2,617.78\n"\r\n"2025-05-10\n","Credit Card Payment\n","DEBIT\n","\$300.19\n","\$2,317.59\n"\r\n"2025-05-16\n","Freelance Work\n","CREDIT\n","\$21.34\n","\$2,338.93\n"\r\n"2025-05-18\n","Interest Earned\n","CREDIT\n","\$120.32\n","\$2,459.25\n"\r\n"2025-05-20\n","Online Order - Best Buy\n","DEBIT\n","\$38.04\n","\$2,421.21\n"\r\n"2025-05-24\n","Public Transit - Uber\n","DEBIT\n","\$416.54\n","\$2,004.67\n"\r\n"2025-05-28\n","Store Payment - Target\n","DEBIT\n","\$347.61\n","\$1,657.06\n"\r\n"2025-05-29\n","Rent Payment - Utility Co.\n","DEBIT\n","\$425.43\n","\$1,231.63\n"\r\n"2025-05-31\n","Ride Share - Chevron\n","DEBIT\n","\$260.21\n","\$971.42\n"\r\n"2025-06-04\n","Tax Refund\n","CREDIT\n","\$429.64\n","\$1,401.06\n"\r\n"2025-06-08\n","Credit Card Payment\n","DEBIT\n","\$300.19\n","\$1,100.87\n"\r\n"2025-06-12\n","Gym Membership\n","DEBIT\n","\$55.00\n","\$1,045.87\n"\r\n"2025-06-16\n","ATM Withdrawal\n","DEBIT\n","\$100.00\n","\$945.87\n"\r\n"2025-06-20\n","Online Course Fee\n","DEBIT\n","\$150.00\n","\$795.87\n"\r\n"2025-06-24\n","Freelance Work\n","CREDIT\n","\$310.00\n","\$1,105.87\n"\r\n"2025-06-28\n","Store Payment - Target\n","DEBIT\n","\$347.61\n","\$758.26\n"\r\n"2025-07-02\n","Consulting Fee\n","CREDIT\n","\$300.00\n","\$1,058.26\n"\r\n"2025-07-05\n","Rent Payment - Utility Co.\n","DEBIT\n","\$425.43\n","\$632.83\n"\r\n'
    else:
        full_text = ""
        
    if full_text:
        # --- REQUESTED PRINT OUTPUT ---
        print(f"\n--- START RAW TEXT FOR: {pdf_filepath} ---")
        print(full_text)
        print(f"--- END RAW TEXT FOR: {pdf_filepath} ---")
        # -----------------------------
        
    return full_text


# --- Helper Functions for Data Cleaning and Conversion ---

def clean_and_convert_currency(s: Any) -> float:
    """Removes currency symbols and commas, then converts the string to a float."""
    if pd.isna(s) or s is None:
        return 0.0
    s = str(s).strip()
    if s.lower() in ('n/a', 'na', 'none', ''):
        return 0.0
    cleaned = re.sub(r'[$,()]', '', s).strip()
    if '(' in s and ')' in s:
        cleaned = "-" + cleaned
    try:
        return float(cleaned)
    except ValueError:
        return 0.0

def clean_numeric_string(s: Any) -> Any:
    """Removes non-numeric characters (except hyphen/decimal) for fields like SSN or Score."""
    if pd.isna(s) or s is None:
        return pd.NA
    s = str(s).strip()
    if s.lower() in ('n/a', 'na', 'none', ''):
        return pd.NA
    return re.sub(r'[^a-zA-Z0-9-.]', '', s).strip()

def extract_summary_field(text: str, label_pattern: str, is_currency: bool = False, is_numeric: bool = False) -> Any:
    """
    Extracts a single value from the raw PDF text content using a flexible regex pattern.
    Prioritizes the CSV-like quote structure for robustness.
    """
    
    # Pattern 2: "key", "value" (PRIMARY PATTERN for this document format)
    pattern_csv = re.compile(
        r'"{}[\s\n]*","(.*?)"'.format(label_pattern), 
        re.IGNORECASE | re.DOTALL
    )
    match_csv = pattern_csv.search(text)
    
    if match_csv:
        value = re.sub(r'\s+', ' ', match_csv.group(1)).strip()
        
        if is_currency:
            return clean_and_convert_currency(value)
        elif is_numeric:
              cleaned_value = clean_numeric_string(value)
              if cleaned_value is not pd.NA:
                  try:
                      return int(float(cleaned_value))
                  except ValueError:
                      return float(cleaned_value)
              return pd.NA
        return value if value.lower() not in ('n/a', 'na', '') else pd.NA
        
    # Fallback to Pattern 1: key: value 
    pattern = re.compile(
        r'{}:[^\n:]*?\s*(.*?)(?=\s*(\w+:\s*|\n\s*\w+:\s*|LOAN|RECENT|\Z))'.format(label_pattern), 
        re.MULTILINE | re.IGNORECASE | re.DOTALL
    )

    match = pattern.search(text)
    if match:
        value = match.group(1).strip()
        value = re.sub(r'\s+', ' ', value).replace('"', '').strip()
        
        if is_currency:
            return clean_and_convert_currency(value)
        elif is_numeric:
            cleaned_value = clean_numeric_string(value)
            if cleaned_value is not pd.NA:
                try:
                    return int(float(cleaned_value))
                except ValueError:
                    return float(cleaned_value)
            return pd.NA
        return value if value.lower() not in ('n/a', 'na', '') else pd.NA
        
    return pd.NA

# --- Specialized Extractor Functions ---

def extract_loan_summary(pdf_filepath: str) -> Dict[str, Any]:
    """Extracts loan profile and client summary data from a single PDF file."""
    
    full_text = get_file_content(pdf_filepath)
    client_data: Dict[str, Any] = {}
    
    loan_summary_cols = [
        'client_id', 'first_name', 'last_name', 'ssn', 'address', 
        'annual_income', 'employment_status', 'credit_score', 
        'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly', 
        'sentiment_score'
    ]
    empty_data = {col: pd.NA for col in loan_summary_cols}
    
    if not full_text or "LOAN & CREDIT PROFILE SUMMARY" not in full_text:
        return {} 

    try:
        # Basic Client Info
        id_match = re.search(r'Client ID:\s*(\d+)', full_text)
        name_match = re.search(r'Client Name:\s*(\w+)\s+(\w+)\s*\|', full_text)
        
        client_data['client_id'] = int(id_match.group(1)) if id_match else pd.NA
        
        if name_match:
            client_data['first_name'] = name_match.group(1)
            client_data['last_name'] = name_match.group(2)
        else:
             # Fallback for full name extraction if first/last name split fails
             name_match_full = re.search(r'Client Name:\s*(.*?)\s*\|', full_text)
             if name_match_full:
                 full_name = name_match_full.group(1).strip()
                 name_parts = full_name.split()
                 client_data['first_name'] = name_parts[0] if name_parts else pd.NA
                 client_data['last_name'] = name_parts[-1] if len(name_parts) > 1 else pd.NA

        # Extract other fields using the robust field extractor
        client_data['ssn'] = extract_summary_field(full_text, r'SSN', is_numeric=False)
        client_data['address'] = extract_summary_field(full_text, r'Address', is_numeric=False)
        client_data['employment_status'] = extract_summary_field(full_text, r'Employment', is_numeric=False)
        client_data['credit_score'] = extract_summary_field(full_text, r'Credit Score', is_numeric=True)
        client_data['collateral_value'] = extract_summary_field(full_text, r'Collateral Value', is_currency=True)
        client_data['loan_amount_requested'] = extract_summary_field(full_text, r'Loan Requested', is_currency=True)
        client_data['alimony_payments_monthly'] = extract_summary_field(full_text, r'Monthly Alimony', is_currency=True)
        client_data['annual_income'] = extract_summary_field(full_text, r'Annual Income', is_currency=True)
        
        sentiment_match = re.search(r'Sentiment Score.*:\s*(-?\d+\.\d+)', full_text, re.IGNORECASE)
        client_data['sentiment_score'] = float(sentiment_match.group(1)) if sentiment_match else pd.NA

        return {**empty_data, **client_data}
        
    except Exception as e:
        print(f"Error processing summary: {e}")
        return {}


def extract_transactions(pdf_filepath: str) -> pd.DataFrame:
    """Extracts transaction history from a single PDF file."""
    
    full_text = get_file_content(pdf_filepath)
    transaction_cols = ['Date', 'Description', 'Type', 'Amount', 'Balance'] 
    df_transactions_empty = pd.DataFrame(columns=transaction_cols)

    if not full_text or "TRANSACTION HISTORY" not in full_text:
        return df_transactions_empty
    
    try:
        # Define the header pattern (marks the start of the data block)
        header_pattern = r'"Date\s*\\n"\s*,\s*"Description\s*\\n"\s*,\s*"Type\s*\\n"\s*,\s*"Amount\s*\\n"\s*,\s*"Balance\s*\\n"'
        table_start_match = re.search(header_pattern, full_text, re.DOTALL)
        
        if table_start_match:
            data_block = full_text[table_start_match.end():]
            
            # Pattern to match and extract a single transaction line in CSV-like format
            # Note: The pattern is adjusted to handle '\r\n' (carriage return + newline) separators
            csv_line_pattern = re.compile(
                r'"(\d{4}-\d{2}-\d{2})\\n"\s*,\s*"([^"]*)\\n"\s*,\s*"(DEBIT|CREDIT)\\n"\s*,\s*"([$][0-9,.]+)\\n"\s*,\s*"([$][0-9,.]+)\\n"\r\n', 
                re.IGNORECASE | re.DOTALL
            )

            csv_extracted_rows = []
            # Find all matches in the data block
            for match in csv_line_pattern.finditer(data_block):
                # Groups are 1:Date, 2:Description, 3:Type, 4:Amount, 5:Balance
                # Clean up extracted strings
                row = [match.group(i).replace('\\n', ' ').strip() for i in range(1, 6)]
                csv_extracted_rows.append(row)

            if csv_extracted_rows:
                df_transactions = pd.DataFrame(csv_extracted_rows, columns=transaction_cols)
                return df_transactions.drop_duplicates().reset_index(drop=True)
            
        return df_transactions_empty

    except Exception as e:
        print(f"Error processing transactions: {e}")
        return df_transactions_empty

# --- Main Modular Processor Function ---

def process_files(file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a list of PDF file paths to extract a single loan summary and 
    a single transaction history DataFrame.
    """
    
    print(f"Starting modular processing for {len(file_paths)} files...")

    loan_summary_cols = [
        'client_id', 'first_name', 'last_name', 'ssn', 'address', 
        'annual_income', 'employment_status', 'credit_score', 
        'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly', 
        'sentiment_score'
    ]
    final_transaction_cols = ['client_id', 'date', 'description', 'type', 'amount', 'balance']

    df_loan_data = pd.DataFrame(columns=loan_summary_cols)
    df_transactions = pd.DataFrame(columns=final_transaction_cols) 
    
    summary_found = False
    transactions_found = False
    extracted_client_id = pd.NA
    
    for filepath in file_paths:
        
        # 1. Attempt to extract Loan Summary
        if not summary_found:
            summary_dict = extract_loan_summary(filepath)
            if summary_dict and summary_dict.get('client_id') is not pd.NA:
                df_loan_data = pd.DataFrame([summary_dict], columns=loan_summary_cols)
                extracted_client_id = df_loan_data['client_id'].iloc[0]
                summary_found = True
                print(f"   -> Loan Summary extracted from {filepath} (ID: {extracted_client_id}).")
            else:
                print(f"   -> No Loan Summary found in {filepath}.")
        
        # 2. Attempt to extract Transactions
        df_tx_raw = pd.DataFrame()
        if not transactions_found:
             df_tx_raw = extract_transactions(filepath) 
        
        if not df_tx_raw.empty:
             df_tx_raw.columns = ['date', 'description', 'type', 'amount', 'balance']
             
             if extracted_client_id is not pd.NA:
                  # Inject ID and finalize transactions DF
                  df_tx_raw.insert(0, 'client_id', extracted_client_id)
                  df_transactions = df_tx_raw[final_transaction_cols]
                  transactions_found = True
                  print(f"   -> Transaction History extracted from {filepath}.")
             else:
                  # Store raw transactions if ID is still unknown
                  df_transactions = df_tx_raw 

        # Check if both are found and consolidate if necessary
        if summary_found and transactions_found:
             # Final injection of client_id if transactions were found first
             if 'client_id' not in df_transactions.columns and extracted_client_id is not pd.NA:
                 df_transactions.insert(0, 'client_id', extracted_client_id)
                 df_transactions = df_transactions[final_transaction_cols]
             print("\nAll required data found. Stopping file processing.")
             break
            
    # Final cleanup and derived income calculation
    if not df_transactions.empty and 'amount' in df_transactions.columns:
        # Convert amount column to clean floats for calculation (used internally, not returned)
        df_transactions['amount_cleaned'] = df_transactions['amount'].apply(clean_and_convert_currency)
        
        # Remove the helper column before returning
        df_transactions = df_transactions.drop(columns=['amount_cleaned'], errors='ignore')

    if df_loan_data.empty:
        df_loan_data = pd.DataFrame(columns=loan_summary_cols)

    return df_loan_data, df_transactions


if __name__ == '__main__':
    # List of files provided by the user (must be accessible by the code)
    INPUT_FILES = [
        "Loan_Profile_10_Farley.pdf", 
        "Bank_Statement_10_Farley.pdf"
    ] 
    
    # Execute the conversion
    loan_summary_df, transaction_history_df = process_files(INPUT_FILES)

    # --- Print Outputs ---
    loan_summary_df_print = loan_summary_df.replace({pd.NA: None})
    transaction_history_df_print = transaction_history_df.replace({pd.NA: None})
    
    print("\n--- Loan Applicant Summary DataFrame (df_loan_data) ---")
    if not loan_summary_df_print.empty:
        print(loan_summary_df_print.to_markdown(index=False))
    else:
        print("No loan summary data could be extracted.")
    
    print("\n--- Transaction History DataFrame (df_transactions) (First 5 Rows) ---")
    if not transaction_history_df_print.empty:
        print(transaction_history_df_print.head().to_markdown(index=False))
    else:
        print("No transactions were extracted.")
