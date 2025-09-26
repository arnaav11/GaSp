import pandas as pd
import re
from typing import List, Tuple, Dict, Any
import os

# --- Helper Functions for Data Cleaning ---

def clean_currency(value: str) -> float:
    """
    Cleans a string value containing currency symbols, commas, and escape characters, 
    converting it to a float.
    """
    if isinstance(value, str):
        # Remove '$', '\', ',', and any leading/trailing whitespace
        cleaned_value = value.strip().replace('$', '').replace(',', '').replace('\\', '')
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
    return ssn.strip().replace('"', '') if isinstance(ssn, str) else ''

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
    Reads content from a list of PDF file paths, parses loan profile and bank
    statement data, and returns two DataFrames.

    NOTE: In a live environment, a library like 'pdfplumber' or 'PyPDF2' would
    be used inside this function to read the text content from the file path.
    Since we cannot install those, we simulate the read operation using the
    pre-extracted text corresponding to the paths.

    Args:
        pdf_file_paths: A list of string paths to the PDF files.

    Returns:
        A tuple containing (loan_applicant_df, bank_transactions_df).
    """

    # --- SIMULATED FILE READING STEP ---
    # This dictionary maps the expected file path to the text content that
    # would be extracted by a real PDF reading library.
    path_to_content_map: Dict[str, str] = {
        "Loan_Profile_10_Farley.pdf": """
            CLIENT LOAN & CREDIT PROFILE
            Client Name: Samuel Farley | Client ID: 10
            Report Date: 2025-09-26
            LOAN & CREDIT PROFILE SUMMARY
            "SSN:", "XXX-XX-9372"
            "Address:", "PSC 4915, Box 3024, APO AE 55468"
            "Credit Score:", "663"
            "Collateral Value:", "N/A"
            "Annual Income:", "\$131,070"
            "Employment:", "Self-Employed"
            "Loan Requested:", "\$26,102"
            "Monthly Alimony:", "N/A"
            Client Sentiment Score: -0.97 (Very Negative - Possible dissatisfaction or risk detected.)
        """,
        "Bank_Statement_10_Farley.pdf": """
            CLIENT ACCOUNT STATEMENT
            Client Name: Samuel Farley | Client ID: 10
            Statement Period: 2025-04-03 to 2025-07-05
            TRANSACTION HISTORY
            Opening Balance: \$3,476.00 | Closing Balance: \$4,283.94
            "Date", "Description", "Type", "Amount", "Balance"
            "2025-04-03", "Online Order - Best Buy", "DEBIT", "\$412.96", "\$3,063.04"
            "2025-04-05", "Streaming Service - Spotify", "DEBIT", "\$244.72", "\$2,818.32"
            "2025-04-08", "Bonus Payment", "CREDIT", "\$71.14", "\$2,889.46"
            "2025-04-10", "POS Debit - Amazon.com", "DEBIT", "\$342.53", "\$2,546.93"
            "2025-04-14", "Gas Station - Uber", "DEBIT", "\$29.29", "\$2,517.64"
            "2025-04-17", "Gift from family", "CREDIT", "\$285.29", "\$2,802.93"
            "2025-04-21", "Coffee Shop - Starbucks", "DEBIT", "\$233.28", "\$2,569.65"
            "2025-04-24", "Refund from Perez, Anderson and Johnson", "CREDIT", "\$425.30", "\$2,994.95"
            "2025-04-27", "Paycheck Deposit", "CREDIT", "\$59.46", "\$3,054.41"
            "2025-04-28", "Paycheck Deposit", "CREDIT", "\$279.13", "\$3,333.54"
            "2025-04-29", "Gas Station - Uber", "DEBIT", "\$323.47", "\$3,010.07"
            "2025-04-30", "Bonus Payment", "CREDIT", "\$125.15", "\$3,135.22"
            "2025-05-04", "Gift from family", "CREDIT", "\$497.96", "\$3,633.18"
            "2025-05-06", "Online Order - Best Buy", "DEBIT", "\$381.01", "\$3,252.17"
            "2025-05-09", "Movie Theater - Spotify", "DEBIT", "\$278.76", "\$2,973.41"
            "2025-05-12", "Gift from family", "CREDIT", "\$174.09", "\$3,147.50"
            "2025-05-13", "POS Debit - Best Buy", "DEBIT", "\$440.03", "\$2,707.47"
            "2025-05-14", "Bonus Payment", "CREDIT", "\$300.19", "\$3,007.66"
            "2025-05-16", "Freelance Work", "CREDIT", "\$21.34", "\$3,029.00"
            "2025-05-18", "Interest Earned", "CREDIT", "\$120.32", "\$3,149.32"
            "2025-05-20", "Online Order - Best Buy", "DEBIT", "\$38.04", "\$3,111.28"
            "2025-05-24", "Public Transit - Uber", "DEBIT", "\$416.54", "\$2,694.74"
            "2025-05-28", "Store Payment - Target", "DEBIT", "\$347.61", "\$2,347.13"
            "2025-05-29", "Rent Payment - Utility Co.", "DEBIT", "\$425.43", "\$1,921.70"
            "2025-05-31", "Ride Share - Chevron", "DEBIT", "\$260.21", "\$1,661.49"
            "2025-06-04", "Tax Refund", "CREDIT", "\$429.64", "\$2,091.13"
            "2025-06-06", "Health Products - Walgreens", "DEBIT", "\$304.53", "\$1,786.60"
            "2025-06-10", "Ride Share - Uber", "DEBIT", "\$103.60", "\$1,683.00"
            "2025-06-11", "Bonus Payment", "CREDIT", "\$200.04", "\$1,883.04"
            "2025-06-15", "Gas Station - Uber", "DEBIT", "\$457.44", "\$1,425.60"
            "2025-06-17", "Gift from family", "CREDIT", "\$128.61", "\$1,554.21"
            "2025-06-20", "Bonus Payment", "CREDIT", "\$115.84", "\$1,670.05"
            "2025-06-24", "Paycheck Deposit", "CREDIT", "\$1,465.55", "\$3,135.60"
            "2025-06-28", "Refund from Hamilton PLC", "CREDIT", "\$80.89", "\$3,216.49"
            "2025-06-30", "Freelance Work", "CREDIT", "\$387.81", "\$3,604.30"
            "2025-07-03", "Gift from family", "CREDIT", "\$421.37", "\$4,025.67"
            "2025-07-04", "Internet Service - Rent Payment", "DEBIT", "\$147.43", "\$3,878.24"
            "2025-07-05", "Bonus Payment", "CREDIT", "\$405.70", "\$4,283.94"
        """
    }
    # --- END SIMULATED FILE READING STEP ---


    loan_applicant_data: List[Dict[str, Any]] = []
    bank_transactions_data: List[Dict[str, Any]] = []

    # Iterate through all provided file paths
    for path in pdf_file_paths:
        # Load content based on the file path
        content = path_to_content_map.get(os.path.basename(path), "")
        if not content:
            print(f"Skipping file: Content for '{path}' was not retrieved. Please ensure the file is accessible and the path is correct.")
            continue

        # --- 1. Extract Common/Client Data ---
        # Client ID
        id_match = re.search(r'Client ID:\s*(\d+)', content)
        client_id = id_match.group(1) if id_match else 'UNKNOWN'

        # Name
        name_line_match = re.search(r'Client Name:\s*.*\|', content)
        if name_line_match:
            first_name, last_name = parse_client_name(name_line_match.group(0))
        else:
            first_name, last_name = '', ''

        # --- 2. Extract Loan Profile Data ---
        if 'LOAN & CREDIT PROFILE SUMMARY' in content:
            record: Dict[str, Any] = {
                'client_id': client_id,
                'first_name': first_name,
                'last_name': last_name
            }

            # SSN, Address, Credit Score, etc.
            data_points = {
                'ssn': r'"SSN:"\s*,\s*"([^"]+)"',
                'address': r'"Address:"\s*,\s*"([^"]+)"',
                'annual_income': r'"Annual Income:"\s*,\s*"([^"]+)"',
                'employment_status': r'"Employment:"\s*,\s*"([^"]+)"',
                'credit_score': r'"Credit Score:"\s*,\s*"([^"]+)"',
                'loan_amount_requested': r'"Loan Requested:"\s*,\s*"([^"]+)"',
                'collateral_value': r'"Collateral Value:"\s*,\s*"([^"]+)"',
                'alimony_payments_monthly': r'"Monthly Alimony:"\s*,\s*"([^"]+)"',
                'sentiment_score': r'Client Sentiment Score:\s*(-?\d+\.?\d*)',
            }

            for key, pattern in data_points.items():
                match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
                value = match.group(1).strip() if match else 'N/A'
                record[key] = value

            # Data Cleaning and Type Conversion for Loan Applicant Data
            record['ssn'] = clean_ssn(record['ssn'])
            record['annual_income'] = clean_currency(record['annual_income'])
            record['credit_score'] = int(record['credit_score']) if record['credit_score'].isdigit() else 0
            record['loan_amount_requested'] = clean_currency(record['loan_amount_requested'])
            record['collateral_value'] = clean_currency(record['collateral_value'])
            record['alimony_payments_monthly'] = clean_currency(record['alimony_payments_monthly'])
            # sentiment_score is a float, checking if it was successfully extracted
            score_str = str(record['sentiment_score'])
            record['sentiment_score'] = float(score_str) if score_str not in ('N/A', 'NA', '') and score_str.replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0

            loan_applicant_data.append(record)

        # --- 3. Extract Bank Statement Transactions Data ---
        if 'TRANSACTION HISTORY' in content:
            # Find the start of the transaction table data (after the header line)
            start_index = content.find('"Date", "Description", "Type", "Amount", "Balance"')
            if start_index == -1:
                print(f"Error: Transaction header not found in {path}.")
                continue

            # Isolate the transaction block
            transaction_block = content[start_index:]
            # Use regex to find all rows of data. This assumes comma-separated, quoted values.
            # We look for a pattern of 5 quoted values in a row.
            # Note: The raw content snippet shows some fields with newlines inside quotes,
            # so we'll use re.DOTALL and clean up the strings later.
            transaction_rows = re.findall(r'"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"', transaction_block, re.DOTALL)

            for row in transaction_rows[1:]: # Skip the header row (index 0)
                if len(row) == 5:
                    date_str, description, type_str, amount_str, balance_str = [s.strip().replace('\n', '') for s in row]
                    bank_transactions_data.append({
                        'client_id': client_id,
                        'date': date_str,
                        'description': description,
                        'type': type_str,
                        'amount': clean_currency(amount_str),
                        'balance': clean_currency(balance_str)
                    })

    # --- 4. Create Final DataFrames ---

    # DF 1: Loan Applicant Data
    loan_applicant_df = pd.DataFrame(loan_applicant_data)
    # Ensure all required columns are present, even if empty
    required_loan_cols = ['client_id', 'first_name', 'last_name', 'ssn', 'address', 'annual_income', 
                          'employment_status', 'credit_score', 'loan_amount_requested', 
                          'collateral_value', 'alimony_payments_monthly', 'sentiment_score']
    for col in required_loan_cols:
        if col not in loan_applicant_df.columns:
            loan_applicant_df[col] = None
    loan_applicant_df = loan_applicant_df[required_loan_cols]

    # DF 2: Bank Transactions Data
    bank_transactions_df = pd.DataFrame(bank_transactions_data)
    bank_transactions_df['date'] = pd.to_datetime(bank_transactions_df['date'], errors='coerce')


    return loan_applicant_df, bank_transactions_df

# --- Example Usage ---

if __name__ == '__main__':
    # List of files to process (using the file names provided in the previous step)
    # Note: In a real system, these would be the absolute file system paths.
    file_list = [
        "Loan_Profile_10_Farley.pdf",
        "Bank_Statement_10_Farley.pdf",
        # You can add more file paths here. The function will try to look up the
        # extracted content for each file name.
    ]

    print("--- Starting Data Extraction ---")
    loan_df, transactions_df = extract_loan_data_to_dfs(file_list)
    print("Extraction complete.\n")

    print("=" * 50)
    print("DataFrame 1: Loan Applicant Data (Structured from Loan_Profile PDF)")
    print("=" * 50)
    print(loan_df.to_string(index=False))

    print("\n--- Data Types for Loan Applicant Data ---")
    print(loan_df.dtypes)


    print("\n" + "=" * 50)
    print("DataFrame 2: Bank Transaction Data (Structured from Bank_Statement PDF)")
    print("=" * 50)
    print(transactions_df.to_string(index=False))

    print("\n--- Data Types for Bank Transaction Data ---")
    print(transactions_df.dtypes)
