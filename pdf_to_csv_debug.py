import pdfplumber
import pandas as pd
import re
import os

def parse_pdf(filepath):
    """
    Extracts structured client information and transaction history from a PDF file.
    Returns two lists: one for client details and one for transactions.
    """
    client_info_data = []
    transaction_data = []
    client_id = None
    
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            # Normalize text by removing bold markers and extra spaces
            clean_text = text.replace("**", "")
            
            # ------------------------
            # Extract Client Report data first as it contains the Client ID
            # ------------------------
            if "Comprehensive Client Financial Report" in clean_text:
                # Extract main client details
                client_match = re.search(r"Client Name:\s*(.+?)\s*\|\s*Client ID:\s*(\d+)", clean_text)
                if client_match:
                    full_name, client_id = client_match.groups()
                    parts = full_name.split()
                    first_name = parts[0]
                    last_name = " ".join(parts[1:]) if len(parts) > 1 else ""

                # Initialize a dictionary for client info
                client_dict = {
                    "client_id": client_id,
                    "first_name": first_name.strip(),
                    "last_name": last_name.strip(),
                }
                
                # Use more specific regex patterns to prevent data bleed between fields
                ssn_match = re.search(r"SSN:\s*([^\n\r]+?)\s*Annual Income:", clean_text)
                address_match = re.search(r"Address:\s*([^\n\r]+?)\s*Employment:", clean_text)
                annual_income_match = re.search(r"Annual Income:\s*\$?([\d,]+)", clean_text)
                employment_status_match = re.search(r"Employment:\s*([^\n\r]+?)\s*Credit Score:", clean_text)
                credit_score_match = re.search(r"Credit Score:\s*(\d+)", clean_text)
                loan_req_match = re.search(r"Loan Requested:\s*\$?([\d,]+)", clean_text)
                collateral_match = re.search(r"Collateral Value:\s*([^\n\r]+?)\s*Monthly Alimony:", clean_text)
                alimony_match = re.search(r"Monthly Alimony:\s*([^\n\r]+?)\s*LOAN & CREDIT PROFILE SUMMARY", clean_text)
                sentiment_match = re.search(r"Client Sentiment Score:\s*(-?[\d.]+)", clean_text)
                
                client_dict["ssn"] = ssn_match.group(1).strip() if ssn_match else None
                client_dict["address"] = address_match.group(1).strip() if address_match else None
                client_dict["annual_income"] = float(annual_income_match.group(1).replace(",", "")) if annual_income_match else None
                client_dict["employment_status"] = employment_status_match.group(1).strip() if employment_status_match else None
                client_dict["credit_score"] = int(credit_score_match.group(1)) if credit_score_match else None
                client_dict["loan_amount_requested"] = float(loan_req_match.group(1).replace(",", "")) if loan_req_match else None
                client_dict["collateral_value"] = collateral_match.group(1).strip() if collateral_match else None
                client_dict["alimony_payments_monthly"] = alimony_match.group(1).strip() if alimony_match else None
                client_dict["sentiment_score"] = float(sentiment_match.group(1)) if sentiment_match else None

                client_info_data.append(client_dict)

            # ------------------------
            # Bank Statement Transactions (using table extraction for reliability)
            # ------------------------
            tables = page.extract_tables()
            for table in tables:
                # Check for the header row to ensure it's the transaction table
                if table and table[0] and "Date" in table[0][0] and "Description" in table[0][1]:
                    # Process the rows, skipping the header
                    for row in table[1:]:
                        if row and row[0]: # Ensure row is not empty
                            date = row[0].strip()
                            desc = row[1].strip()
                            ttype = row[2].strip()
                            
                            # Clean up and convert amount/balance
                            amt_str = row[3].replace("$", "").replace(",", "")
                            bal_str = row[4].replace("$", "").replace(",", "")
                            
                            try:
                                amount = float(amt_str) * (-1 if ttype == "DEBIT" else 1)
                                balance = float(bal_str)
                            except (ValueError, IndexError):
                                # Skip row if amount or balance is missing or malformed
                                continue

                            transaction_dict = {
                                "client_id": client_id,
                                "date": date,
                                "description": desc,
                                "type": ttype,
                                "amount": amount,
                                "balance": balance,
                            }
                            transaction_data.append(transaction_dict)

    return client_info_data, transaction_data


def process_all_pdfs(
    pdf_files=None,
    output_folder="output"
):
    """Processes all given PDFs into two CSV files."""
    if pdf_files is None:
        # Example files, replace with your actual file paths
        pdf_files = ["Client_Report_1_Newman.pdf"]

    all_client_info = []
    all_transactions = []
    
    for filepath in pdf_files:
        if os.path.exists(filepath):
            client_data, transaction_data = parse_pdf(filepath)
            all_client_info.extend(client_data)
            all_transactions.extend(transaction_data)
        else:
            print(f"⚠️ Skipping missing file: {filepath}")

    # ✅ Left-align all rows (shift non-empty values to the left)
    df = df.apply(lambda row: pd.Series([x for x in row if pd.notna(x)]), axis=1)

    # ✅ Optional: clean Description values
    if "Description" in df.columns:
        df["Description"] = (
            df["Description"]
            .fillna("")
            .astype(str)
            .str.replace(".pdf", "", regex=False)
            .str.strip(",")
        )

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Create and save client info CSV
    if all_client_info:
        df_client_info = pd.DataFrame(all_client_info)
        # Reorder columns as requested
        cols = ["client_id", "first_name", "last_name", "ssn", "address", "annual_income", 
                "employment_status", "credit_score", "loan_amount_requested", 
                "collateral_value", "alimony_payments_monthly", "sentiment_score"]
        df_client_info = df_client_info.reindex(columns=cols)
        
        output_file_info = os.path.join(output_folder, "client_info.csv")
        df_client_info.to_csv(output_file_info, index=False)
        print(f"✅ Client Info CSV created: {output_file_info} ({len(df_client_info)} rows)")

<<<<<<< HEAD
    # Create and save transactions CSV
    if all_transactions:
        df_transactions = pd.DataFrame(all_transactions)
        # Reorder columns as requested
        cols = ["client_id", "date", "description", "type", "amount", "balance"]
        df_transactions = df_transactions.reindex(columns=cols)
        output_file_transactions = os.path.join(output_folder, "client_transactions.csv")
        df_transactions.to_csv(output_file_transactions, index=False)
        print(f"✅ Client Transactions CSV created: {output_file_transactions} ({len(df_transactions)} rows)")
        
    return df_client_info, df_transactions


# Run the script with your PDF file
# process_all_pdfs(pdf_files=["Client_Report_1_Newman.pdf"])
=======
    output_file = os.path.join(output_folder, "Master_All_Clients.csv")
    df.to_csv(output_file, index=False, header=False)  # no weird headers, pure data
    print(f"✅ Master CSV created (left-aligned): {output_file} ({len(df)} rows)")


# Run
process_all_pdfs()
>>>>>>> 455116060e98c7b5aa22eb5a8d7c16b1984a07c8
