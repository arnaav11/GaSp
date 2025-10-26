import pandas as pd
import re
from typing import List, Tuple, Dict, Any
import os
import fitz  # PyMuPDF library
import matplotlib.pyplot as plt
import seaborn as sns

# --- Helper Functions for Data Cleaning (from pdf_to_csv_debug.py) ---

def clean_currency(value: str) -> float:
    """Cleans a string value containing currency symbols, commas, and newlines."""
    if isinstance(value, str):
        cleaned_value = value.strip().replace('$', '').replace(',', '').replace('\n', '')
        if cleaned_value.upper() in ('N/A', 'NA', ''):
            return 0.0
        try:
            return float(cleaned_value)
        except ValueError:
            return 0.0
    return float(value) if value is not None else 0.0

def clean_ssn(ssn: str) -> str:
    """Cleans SSN format."""
    return ssn.strip().replace('"', '').replace('\n', '') if isinstance(ssn, str) else ''

def parse_client_name(client_name_line: str) -> Tuple[str, str]:
    """Extracts first and last name from a 'Client Name: First Last' string."""
    try:
        match = re.search(r'Client Name:\s*(\w+)\s*(\w+)\s*\|', client_name_line)
        if match:
            return match.group(1), match.group(2)
        name_part = client_name_line.split('|')[0].replace('Client Name:', '').strip()
        parts = name_part.split()
        return parts[0], parts[-1]
    except Exception:
        return '', ''

# --- PDF Data Extraction Function (from pdf_to_csv_debug.py) ---

def extract_loan_data_to_dfs(pdf_file_paths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads and parses data from a list of PDF file paths."""
    loan_applicant_data = []
    bank_transactions_data = []
    for path in pdf_file_paths:
        try:
            doc = fitz.open(path)
            content = "".join(page.get_text() for page in doc)
            doc.close()
        except Exception as e:
            print(f"Error reading '{path}': {e}. Skipping.")
            continue
        if not content:
            continue

        id_match = re.search(r'Client ID:\s*(\d+)', content)
        client_id = id_match.group(1) if id_match else 'UNKNOWN'
        name_line_match = re.search(r'Client Name:\s*.*\|', content)
        first_name, last_name = parse_client_name(name_line_match.group(0)) if name_line_match else ('', '')

        if 'LOAN & CREDIT PROFILE SUMMARY' in content:
            record = {'client_id': client_id, 'first_name': first_name, 'last_name': last_name}
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
                record[key] = match.group(1).strip() if match else 'N/A'
            
            record['ssn'] = clean_ssn(record.get('ssn', ''))
            record['annual_income'] = clean_currency(record.get('annual_income', '0'))
            record['credit_score'] = int(clean_currency(record.get('credit_score', '0')))
            record['loan_amount_requested'] = clean_currency(record.get('loan_amount_requested', '0'))
            record['collateral_value'] = clean_currency(record.get('collateral_value', '0'))
            record['alimony_payments_monthly'] = clean_currency(record.get('alimony_payments_monthly', '0'))
            score_str = str(record.get('sentiment_score', '0'))
            record['sentiment_score'] = float(score_str) if score_str.replace('.', '', 1).replace('-', '', 1).isdigit() else 0.0
            loan_applicant_data.append(record)

        if 'TRANSACTION HISTORY' in content:
            transaction_block_match = re.search(r'TRANSACTION HISTORY\s*(.*)', content, re.DOTALL)
            if transaction_block_match:
                transaction_block = transaction_block_match.group(1)
                transaction_rows = re.findall(
                    r'^(\d{4}-\d{2}-\d{2})\s+(.+?)\s+(CREDIT|DEBIT)\s+([\$\d,\.]+)\s+([\$\d,\.]+)$',
                    transaction_block,
                    re.MULTILINE
                )
                for row in transaction_rows:
                    date_str, description, type_str, amount_str, balance_str = row
                    bank_transactions_data.append({
                        'client_id': client_id, 'date': date_str.strip(), 'description': description.strip(),
                        'type': type_str.strip(), 'amount': clean_currency(amount_str),
                        'balance': clean_currency(balance_str)
                    })
    
    loan_df = pd.DataFrame(loan_applicant_data)
    trans_df = pd.DataFrame(bank_transactions_data)
    if not trans_df.empty:
        trans_df['date'] = pd.to_datetime(trans_df['date'], errors='coerce')

    return loan_df, trans_df

# --- Pipeline Step Functions ---

def step_1_data_receiver(filepaths: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Acts as the initial data handler, extracting data from uploaded PDF files.
    """
    print("\n[STEP 1/3] Data received and initialized.")
    print(f"  -> Processing files: {[os.path.basename(p) for p in filepaths]}")
    return extract_loan_data_to_dfs(filepaths)

def step_2_analyze(df_client_info: pd.DataFrame, df_transactions: pd.DataFrame) -> Dict[str, Any]:
    """
    Performs a more detailed, multi-factor client validity analysis and generates results for the UI.
    """
    print("\n[STEP 2/3] Running client validity analysis...")
    print(f"  -> Analyzing {len(df_client_info)} clients with {len(df_transactions)} transactions...")
    
    # Use the first client's data for the analysis report
    client_data = df_client_info.iloc[0]
    
    # Extract key metrics from the data
    credit_score = int(client_data.get('credit_score', 300))
    annual_salary = float(client_data.get('annual_income', 0))
    loan_amount_requested = float(client_data.get('loan_amount_requested', 0))
    alimony_payments_monthly = float(client_data.get('alimony_payments_monthly', 0))
    sentiment_score = float(client_data.get('sentiment_score', 0))
    total_debit = 0.0
    if not df_transactions.empty:
        total_debit = df_transactions[df_transactions['type'] == 'DEBIT']['amount'].sum()

    # --- Enhanced Analysis Logic ---
    
    # 1. Calculate a dynamic Debt-to-Income (DTI) ratio
    monthly_income = annual_salary / 12 if annual_salary > 0 else 1
    # Estimate monthly payment on new loan (e.g., 5-year term) + existing alimony
    estimated_new_debt_monthly = (loan_amount_requested / 60) 
    total_monthly_debt = estimated_new_debt_monthly + alimony_payments_monthly
    dti_ratio = total_monthly_debt / monthly_income if monthly_income > 0 else 1.0

    # 2. Multi-factor risk scoring
    risk_score = 0
    risk_factors = []

    if credit_score < 650:
        risk_score += 40
        risk_factors.append("a low credit score")
    elif credit_score < 740:
        risk_score += 15
        risk_factors.append("a fair credit score")

    if dti_ratio > 0.43:
        risk_score += 40
        risk_factors.append("a high debt-to-income ratio")
    elif dti_ratio > 0.36:
        risk_score += 20
        risk_factors.append("a moderate debt-to-income ratio")

    if annual_salary < 45000:
        risk_score += 10
        risk_factors.append("a lower annual income")
        
    if sentiment_score < 0:
        risk_score += 15
        risk_factors.append("negative sentiment detected in documents")

    # 3. Map final risk score to outputs
    if risk_score > 50:
        fraud = "Medium"
        viability = "Low"
        approval = "Denied"
        reasons = " and ".join(risk_factors)
        insights = f"Client presents a higher risk due to {reasons}. Not recommended for approval at this time."
    elif risk_score > 20:
        fraud = "Low"
        viability = "Medium"
        approval = "Conditional Approval"
        reasons = " and ".join(risk_factors)
        insights = f"Client has a fair profile but approval is conditional due to {reasons}. Further review is recommended."
    else:
        fraud = "Low"
        viability = "High"
        approval = "Approved"
        insights = "Client has an excellent credit history and a strong financial profile. Low risk for investment."
        
    print("  -> Analysis complete.")

    # Compile results into a dictionary for the UI
    return {
        'credit_score': credit_score,
        'fraud': fraud,
        'viability': viability,
        'dti': f"{dti_ratio:.1%}",
        'annual_salary': annual_salary,
        'total_debit': total_debit,
        'approval': approval,
        'insights': insights,
        'client_data': client_data, # Pass along for visual generation
        'transactions_df': df_transactions, # Pass along for visual generation
    }

def step_3_generate_visuals(analysis_results: Dict, output_path: str = 'temp_uploaded_files') -> str:
    """Generates and saves a visual summary of the client's finances."""
    print("\n[STEP 3/3] Generating visual report...")
    
    transactions_df = analysis_results['transactions_df']
    client_data = analysis_results['client_data']
    
    if transactions_df.empty:
        print(" -> No transaction data to visualize.")
        return ""

    # Set theme for the plot
    sns.set_theme(style="whitegrid", rc={"axes.facecolor": "#121212", "grid.color": "#2a2a2a", 
                                        "text.color": "white", "xtick.color": "white", 
                                        "ytick.color": "white", "axes.labelcolor": "white",
                                        "figure.facecolor": "#000000"})

    fig, ax = plt.subplots(figsize=(10, 5))

    # Summarize transactions by month
    transactions_df['month'] = transactions_df['date'].dt.to_period('M')
    monthly_summary = transactions_df.groupby(['month', 'type'])['amount'].sum().unstack().fillna(0)
    
    monthly_summary.plot(kind='bar', ax=ax, color={"CREDIT": "#b19cd9", "DEBIT": "#555555"})

    ax.set_title(f"Monthly Credits vs. Debits for {client_data['first_name']} {client_data['last_name']}", color="#b19cd9", fontsize=16)
    ax.set_xlabel("Month", color="white")
    ax.set_ylabel("Amount ($)", color="white")
    ax.tick_params(axis='x', rotation=45)
    ax.legend(title="Transaction Type")
    plt.tight_layout()

    # Save the plot to the specified output path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    chart_path = os.path.join(output_path, f"financial_summary_{client_data['client_id']}.png")
    plt.savefig(chart_path, transparent=False, facecolor='#000000')
    plt.close(fig)
    print(f" -> Visual report saved to: {chart_path}")
    return chart_path


# --- Main Pipeline Function (Streamlit Entry Point) ---

def run_gasp_pipeline(file_paths: List[str]) -> Dict[str, Any]:
    """
    The main callable function for the Streamlit application.
    It runs the entire analysis pipeline, including visual generation.
    """
    pipeline_summary = []
    print("\nThank you for choosing GA$P. We are processing your request...")
    pipeline_summary.append("SETUP: All custom modules imported successfully.")
    pipeline_summary.append("\nThank you for choosing GA$P. We are processing your request...")
    
    try:
        # 1. Initialize Data
        df_info, df_trans = step_1_data_receiver(file_paths)
        pipeline_summary.extend([
            "\n[STEP 1/3] Data received and initialized.",
            f" -> Processing files: {[os.path.basename(p) for p in file_paths]}"
        ])
        
        if df_info.empty:
            return {
                "pipeline_summary": pipeline_summary + ["[ERROR] No loan profile data could be extracted."],
                "error": "Could not parse loan profile PDF. Please check the file format and content."
            }

        # 2. Analyze Data
        analysis_results = step_2_analyze(df_info, df_trans)
        pipeline_summary.extend([
            "\n[STEP 2/3] Running client validity analysis...",
            f" -> Analyzing {len(df_info)} clients with {len(df_trans)} transactions...",
            " -> Analysis complete."
        ])
        
        # 3. Generate Visuals
        chart_path = step_3_generate_visuals(analysis_results)
        analysis_results['chart_path'] = chart_path

        # Clean up dataframes from dict before returning to UI
        del analysis_results['client_data']
        del analysis_results['transactions_df']
        
        # 4. Finalize report for UI
        print("\n[STEP 4/4] Finalizing report...")
        pipeline_summary.append("\n[STEP 3/3] Generating final report...") # This line is kept for consistency in logs
        analysis_results['pipeline_summary'] = pipeline_summary
        print("\nGA$P process successfully completed.")
        
    except Exception as e:
        print(f"\nFATAL ERROR encountered during pipeline execution: {e}")
        return {"error": str(e), "pipeline_summary": pipeline_summary}
        
    return analysis_results

