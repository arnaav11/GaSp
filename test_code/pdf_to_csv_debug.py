import pandas as pd
import re
from typing import List, Tuple, Dict, Any
import os
import fitz  # PyMuPDF library

# --- Helper Functions for Data Cleaning ---

def clean_currency(value: str) -> float:
    """
    Cleans a string value containing currency symbols, commas, and escape characters, 
    converting it to a float.
    """
    if isinstance(value, str):
        # Remove '$', ',', any newlines, and leading/trailing whitespace
        cleaned_value = value.strip().replace('$', '').replace(',', '').replace('\n', '')
        if cleaned_value.upper() in ('N/A', 'NA', ''):
            return 0.0
        try:
            return float(cleaned_value)
        except ValueError:
            print(f"Warning: Could not convert '{value}' to float.")
            return 0.0
    return float(value) if value is not None else 0.0

def clean_ssn(ssn: str) -> str:
    """Cleans SSN format to be consistent or return a placeholder."""
    return ssn.strip().replace('"', '').replace('\n', '') if isinstance(ssn, str) else ''

def parse_client_name(client_name_line: str) -> Tuple[str, str]:
    """Extracts first and last name from a 'Client Name: First Last' string."""
    try:
        match = re.search(r'Client Name:\s*(\w+)\s*(\w+)\s*\|', client_name_line)
        if match:
            return match.group(1), match.group(2)
        # Fallback parsing
        name_part = client_name_line.split('|')[0].replace('Client Name:', '').strip()
        parts = name_part.split()
        return parts[0], parts[-1]
    except Exception:
        return '', ''

# --- Main Extraction Logic ---

def extract_loan_data_to_dfs(pdf_file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reads text content from a list of PDF file paths using PyMuPDF, parses loan
    profile and bank statement data, and returns two DataFrames.

    Args:
        pdf_file_paths: A list of string paths to the PDF files.

    Returns:
        A tuple containing (loan_applicant_df, bank_transactions_df).
    """
    loan_applicant_data: List[Dict[str, Any]] = []
    bank_transactions_data: List[Dict[str, Any]] = []

    for path in pdf_file_paths:
        try:
            doc = fitz.open(path)
            content = ""
            for page in doc:
                content += page.get_text()
            doc.close()
        except Exception as e:
            print(f"Error reading or opening '{path}': {e}. Skipping file.")
            continue

        if not content:
            print(f"Warning: No text content extracted from '{path}'. Skipping file.")
            continue
        
        # --- 1. Extract Common/Client Data ---
        id_match = re.search(r'Client ID:\s*(\d+)', content)
        client_id = id_match.group(1) if id_match else 'UNKNOWN'

        name_line_match = re.search(r'Client Name:\s*.*\|', content)
        first_name, last_name = parse_client_name(name_line_match.group(0)) if name_line_match else ('', '')

        # --- 2. Extract Loan Profile Data ---
        if 'LOAN & CREDIT PROFILE SUMMARY' in content:
            record: Dict[str, Any] = {'client_id': client_id, 'first_name': first_name, 'last_name': last_name}
            # Regex patterns updated to be more flexible and handle multiline values.
            data_points = {
                'ssn': r'SSN:\s*([^\n]+)',
                'address': r'Address:\s*([^\n]+)',
                'annual_income': r'Annual Income:\s*([^\n]+)',
                'employment_status': r'Employment:\s*([^\n]+)',
                'credit_score': r'Credit Score:\s*([^\n]+)',
                'loan_amount_requested': r'Loan Requested:\s*([^\n]+)',
                'collateral_value': r'Collateral Value:\s*([^\n]+)',
                'alimony_payments_monthly': r'Monthly Alimony:\s*([^\n]+)',
                'sentiment_score': r'Client Sentiment Score:\s*(-?\d+\.?\d*)',
            }

            for key, pattern in data_points.items():
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                value = match.group(1).strip() if match else 'N/A'
                record[key] = value

            # Data Cleaning and Type Conversion
            record['ssn'] = clean_ssn(record.get('ssn', ''))
            record['annual_income'] = clean_currency(record.get('annual_income', '0'))
            record['credit_score'] = int(clean_currency(record['credit_score'])) if record.get('credit_score', '0').replace('.', '', 1).isdigit() else 0
            record['loan_amount_requested'] = clean_currency(record.get('loan_amount_requested', '0'))
            record['collateral_value'] = clean_currency(record.get('collateral_value', '0'))
            record['alimony_payments_monthly'] = clean_currency(record.get('alimony_payments_monthly', '0'))
            score_str = str(record.get('sentiment_score', '0'))
            record['sentiment_score'] = float(score_str) if score_str not in ('N/A', 'NA', '') and score_str.replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0

            loan_applicant_data.append(record)

        # --- 3. Extract Bank Statement Transactions Data ---
        if 'TRANSACTION HISTORY' in content:
            # Find the block of text containing transactions
            transaction_block_match = re.search(r'TRANSACTION HISTORY\s*(.*)', content, re.DOTALL)
            if transaction_block_match:
                transaction_block = transaction_block_match.group(1)
                # Regex to find transaction lines that are not quoted
                # Format: DATE, DESCRIPTION (can have spaces), TYPE, AMOUNT, BALANCE
                transaction_rows = re.findall(
                    r'^(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(CREDIT|DEBIT)\s+([\$\d,\.]+)\s+([\$\d,\.]+)$',
                    transaction_block,
                    re.MULTILINE
                )
                
                for row in transaction_rows:
                    if len(row) == 5:
                        date_str, description, type_str, amount_str, balance_str = row
                        bank_transactions_data.append({
                            'client_id': client_id,
                            'date': date_str.strip(),
                            'description': description.strip(),
                            'type': type_str.strip(),
                            'amount': clean_currency(amount_str),
                            'balance': clean_currency(balance_str)
                        })

    # --- 4. Create Final DataFrames ---
    loan_applicant_df = pd.DataFrame(loan_applicant_data)
    required_loan_cols = ['client_id', 'first_name', 'last_name', 'ssn', 'address', 'annual_income', 
                          'employment_status', 'credit_score', 'loan_amount_requested', 
                          'collateral_value', 'alimony_payments_monthly', 'sentiment_score']
    for col in required_loan_cols:
        if col not in loan_applicant_df.columns:
            loan_applicant_df[col] = None
    loan_applicant_df = loan_applicant_df[required_loan_cols]

    bank_transactions_df = pd.DataFrame(bank_transactions_data)
    if not bank_transactions_df.empty:
        bank_transactions_df['date'] = pd.to_datetime(bank_transactions_df['date'], errors='coerce')

    return loan_applicant_df, bank_transactions_df

# --- Example Usage ---

if __name__ == '__main__':
    # List of files to process. Assumes these files exist.
    file_list = [
        "test_code/Loan_Profile_9_Wright.pdf",
        "test_code/Bank_Statement_9_Wright.pdf",
    ]

    print("--- Starting Data Extraction ---")
    loan_df, transactions_df = extract_loan_data_to_dfs(file_list)
    print("Extraction complete.\n")

    print("=" * 50)
    print("DataFrame 1: Loan Applicant Data")
    print("=" * 50)
    if not loan_df.empty:
        print(loan_df.to_string(index=False))
        print("\n--- Data Types for Loan Applicant Data ---")
        print(loan_df.dtypes)
    else:
        print("No loan applicant data extracted.")

    print("\n" + "=" * 50)
    print("DataFrame 2: Bank Transaction Data")
    print("=" * 50)
    if not transactions_df.empty:
        print(transactions_df.to_string(index=False))
        print("\n--- Data Types for Bank Transaction Data ---")
        print(transactions_df.dtypes)
    else:
        print("No bank transaction data extracted.")

