import os
import pandas as pd
import re

# Transaction regex pattern (matches lines like your example)
transaction_pattern = re.compile(
    r"(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(DEBIT|CREDIT)\s+\$?([\d,]+\.\d{2})\s+\$?([\d,]+\.\d{2})"
)

# Map of clients to their reports
clients = {
    "Newman": "Client_Report_1_Newman.pdf",
    "Martinez": "Client_Report_2_Martinez.pdf",
    "Henderson": "Client_Report_3_Henderson.pdf",
    "Brock": "Client_Report_4_Brock.pdf",
}

def extract_transactions(pdf_path):
    """Extract transaction lines from a PDF into a DataFrame-like structure"""
    import pdfplumber
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for line in text.splitlines():
                m = transaction_pattern.match(line.strip())
                if m:
                    date, desc, ttype, amount, balance = m.groups()
                    rows.append([date, desc, ttype, amount, balance])
    return rows

all_transactions = []

# Process each client
for client, pdf in clients.items():
    if not os.path.exists(pdf):
        print(f"⚠️ File missing: {pdf}")
        continue

    rows = extract_transactions(pdf)

    if rows:
        df = pd.DataFrame(rows, columns=["Date", "Description", "Type", "Amount", "Balance"])
        df.insert(0, "Client", client)  # add client name column
        out_csv = f"transactions_{client}.csv"
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved {out_csv}")
        all_transactions.append(df)
    else:
        print(f"ℹ️ No transactions found for {client}")

# Combine all into one
if all_transactions:
    combined = pd.concat(all_transactions, ignore_index=True)
    combined.to_csv("transactions_all.csv", index=False)
    print("✅ Saved combined CSV: transactions_all.csv")
