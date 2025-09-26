import pdfplumber
import pandas as pd
import re
import glob
import os

def parse_pdf(filepath):
    """Extract structured data from one PDF file."""
    data = []
    with pdfplumber.open(filepath) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if not text:
                continue

            # Normalize by removing **bold markers**
            clean_text = text.replace("**", "")

            # ------------------------
            # Bank Statement Transactions
            # ------------------------
            if "Monthly Account Statement" in clean_text:
                transaction_lines = re.findall(
                    r"(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(DEBIT|CREDIT)\s+\$([\d,]+\.\d{2})\s+\$([\d,]+\.\d{2})",
                    clean_text
                )
                for t in transaction_lines:
                    date, desc, ttype, amt, bal = t
                    data.append({
                        "Date": date,
                        "Description": desc.strip(),
                        "Type": ttype,
                        "Amount": float(amt.replace(",", "")) * (-1 if ttype == "DEBIT" else 1),
                        "Balance": float(bal.replace(",", "")),
                        "File": os.path.basename(filepath),
                        "Source": "Bank Statement"
                    })

            # ------------------------
            # Client Reports
            # ------------------------
            if "Comprehensive Client Financial Report" in clean_text:
                client_match = re.search(r"Client Name:\s*(.+?)\s*\|\s*Client ID:\s*(\d+)", clean_text)
                if client_match:
                    client_name, client_id = client_match.groups()
                    data.append({
                        "Client ID": client_id,
                        "Client Name": client_name,
                        "File": os.path.basename(filepath),
                        "Source": "Client Report"
                    })

                income_match = re.search(r"Annual Income:\s*\$?([\d,]+)", clean_text)
                if income_match:
                    data.append({
                        "Annual Income": float(income_match.group(1).replace(",", "")),
                        "File": os.path.basename(filepath),
                        "Source": "Client Report"
                    })

                credit_match = re.search(r"Credit Score:\s*(\d+)", clean_text)
                if credit_match:
                    data.append({
                        "Credit Score": int(credit_match.group(1)),
                        "File": os.path.basename(filepath),
                        "Source": "Client Report"
                    })

                loan_req_match = re.search(r"Loan Requested:\s*\$?([\d,]+)", clean_text)
                if loan_req_match:
                    data.append({
                        "Loan Requested": float(loan_req_match.group(1).replace(",", "")),
                        "File": os.path.basename(filepath),
                        "Source": "Client Report"
                    })

            # ------------------------
            # Loan Applications
            # ------------------------
            if "Personal Loan Application Summary" in clean_text:
                fullname_match = re.search(r"Full Name:\s*(.+)", clean_text)
                if fullname_match:
                    data.append({
                        "Client Name": fullname_match.group(1).strip(),
                        "File": os.path.basename(filepath),
                        "Source": "Loan Application"
                    })

                income_match = re.search(r"Annual Income:\s*\$?([\d,]+)", clean_text)
                if income_match:
                    data.append({
                        "Annual Income": float(income_match.group(1).replace(",", "")),
                        "File": os.path.basename(filepath),
                        "Source": "Loan Application"
                    })

                credit_match = re.search(r"Credit Score:\s*(\d+)", clean_text)
                if credit_match:
                    data.append({
                        "Credit Score": int(credit_match.group(1)),
                        "File": os.path.basename(filepath),
                        "Source": "Loan Application"
                    })

                loan_req_match = re.search(r"Loan Amount Requested:\s*\$?([\d,]+)", clean_text)
                if loan_req_match:
                    data.append({
                        "Loan Amount Requested": float(loan_req_match.group(1).replace(",", "")),
                        "File": os.path.basename(filepath),
                        "Source": "Loan Application"
                    })

    return data


def process_all_pdfs(
    folder=r"C:\Users\singh\OneDrive\Desktop\pdfs",
    output_folder=r"C:\Users\singh\GaSp\test\databases"
):
    """Process all PDFs in a folder into one CSV at the given output path."""
    all_data = []
    for filepath in glob.glob(os.path.join(folder, "*.pdf")):
        all_data.extend(parse_pdf(filepath))

    df = pd.DataFrame(all_data)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    output_file = os.path.join(output_folder, "Master_All_Clients.csv")
    df.to_csv(output_file, index=False)
    print(f"✅ Master CSV created: {output_file} ({len(df)} rows)")


# Run
process_all_pdfs()