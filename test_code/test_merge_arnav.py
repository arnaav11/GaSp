import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime
import re # <-- ADDED: Need 're' for robust string cleaning

# --- 1. PDF Setup ---
styles = getSampleStyleSheet()
NAVY_BLUE = colors.HexColor('#003366')
LIGHT_GREY = colors.HexColor('#F0F0F0')
WHITE = colors.white
BLACK = colors.black

# Define shared styles
styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, spaceAfter=15, alignment=0, textColor=NAVY_BLUE))
styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold', fontSize=12, spaceAfter=5, textColor=NAVY_BLUE, backColor=LIGHT_GREY))
styles.add(ParagraphStyle(name='ProfileBody', fontName='Helvetica', fontSize=10, spaceAfter=5, leading=12))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica-Oblique', fontSize=8, spaceAfter=5, textColor=colors.grey))

# Style for the final assessment paragraph
assessment_style = ParagraphStyle(name='Assessment', fontName='Helvetica-Bold', fontSize=11, spaceAfter=15, textColor=NAVY_BLUE)


# ----------------------------------------------------------------------
## HELPER FUNCTION: CURRENCY CONVERSION
# ----------------------------------------------------------------------
def clean_and_convert_currency(s) -> float:
    """Removes currency symbols and commas, then converts the string to a float."""
    if pd.isna(s) or s is None:
        return 0.0
    
    s = str(s).strip()
    if s.lower() in ('n/a', 'na', 'none', ''):
        return 0.0

    # Remove currency symbols ($,), commas, and handle parentheses for negative numbers
    cleaned = re.sub(r'[$,]', '', s).strip()
    
    # Check for parentheses which often denote negative numbers in finance
    if '(' in s and ')' in s:
        cleaned = "-" + re.sub(r'[()]', '', cleaned)
        
    try:
        return float(cleaned)
    except ValueError:
        # Fallback in case the original string was already a valid float but was typecast as str
        try:
            return float(s)
        except ValueError:
            return 0.0


# ----------------------------------------------------------------------
## NEW FUNCTION: STATEMENT PDF GENERATION
# ----------------------------------------------------------------------

def create_client_statement_pdf(p: pd.Series, client_transactions: pd.DataFrame, client_id: int, pdf_filename: str):
    """
    Generates a standalone Bank Statement PDF for a single client.
    
    Args:
        p: The client's profile data (Series).
        client_transactions: The client's sorted transaction data (DataFrame).
        client_id: The client's ID.
        pdf_filename: The full path and filename for the output PDF.
    """
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # --- Document Header ---
    story.append(Paragraph("CLIENT ACCOUNT STATEMENT", styles['TitleStyle']))
    story.append(Paragraph(f"Client Name: {p.get('first_name', 'N/A')} {p.get('last_name', 'N/A')} | Client ID: {client_id}", styles['SectionHeader']))
    
    if not client_transactions.empty:
        # Determine the statement period
        start_date = client_transactions['date'].min().strftime('%Y-%m-%d')
        end_date = client_transactions['date'].max().strftime('%Y-%m-%d')
        
        story.append(Paragraph(f"Statement Period: {start_date} to {end_date}", styles['SmallText']))
        story.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", styles['SmallText']))
        story.append(Spacer(1, 0.25 * inch))

        # ----------------------------------------------------------------------
        ## TRANSACTION HISTORY TABLE
        # ----------------------------------------------------------------------
        story.append(Paragraph("TRANSACTION HISTORY", styles['SectionHeader']))
        
        # Prepare transaction data for table (converting date back to string for display)
        transaction_data_list = client_transactions[['date', 'description', 'type', 'amount_fmt', 'balance_fmt']].copy()
        transaction_data_list['date'] = transaction_data_list['date'].dt.strftime('%Y-%m-%d')
        transaction_data_list = transaction_data_list.values.tolist()
        
        table_header = ['Date', 'Description', 'Type', 'Amount', 'Balance']
        
        # Determine the opening/closing balance for the statement period
        try:
            first_transaction = client_transactions.iloc[0]
            # This calculation now works because 'amount' and 'balance' are floats
            opening_balance = first_transaction['balance'] - first_transaction['amount'] if first_transaction['type'] == 'CREDIT' else first_transaction['balance'] + first_transaction['amount']
            closing_balance = client_transactions['balance_fmt'].iloc[-1]
        except Exception:
            opening_balance = 0.00
            closing_balance = "$0.00"
        
        story.append(Paragraph(f"Opening Balance: ${opening_balance:,.2f} | Closing Balance: {closing_balance}", styles['ProfileBody']))
        story.append(Spacer(1, 0.1 * inch))

        # Create the transaction table
        transaction_table_data = [table_header] + transaction_data_list
        col_widths = [0.8*inch, 3.0*inch, 0.7*inch, 1.5*inch, 1.5*inch] 

        transaction_style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), NAVY_BLUE),
            ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'), # Right align Amount and Balance
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
            ('GRID', (0, 0), (-1, -1), 0.5, BLACK),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ])

        transaction_table = Table(transaction_table_data, colWidths=col_widths)
        transaction_table.setStyle(transaction_style)
        story.append(transaction_table)
    else:
        story.append(Paragraph("No recent transaction history available for this client.", styles['ProfileBody']))

    # --- Build the PDF ---
    doc.build(story)


# ----------------------------------------------------------------------
## MODIFIED FUNCTION: LOAN PROFILE PDF GENERATION
# ----------------------------------------------------------------------

def create_client_loan_report_pdf(p: pd.Series, client_id: int, pdf_filename: str):
    """
    Generates a standalone Loan & Credit Profile Summary PDF for a single client.
    
    Args:
        p: The client's profile data (Series).
        client_id: The client's ID.
        pdf_filename: The full path and filename for the output PDF.
    """
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # --- Document Header ---
    story.append(Paragraph("CLIENT LOAN & CREDIT PROFILE", styles['TitleStyle']))
    story.append(Paragraph(f"Client Name: {p.get('first_name', 'N/A')} {p.get('last_name', 'N/A')} | Client ID: {client_id}", styles['SectionHeader']))
    story.append(Paragraph(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d')}", styles['SmallText']))
    
    # Display the final predictive assessment
    story.append(Paragraph(f"Final Assessment: {p.get('assessment', 'N/A')}", assessment_style))

    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------------
    ## LOAN & CREDIT PROFILE SUMMARY
    # ----------------------------------------------------------------------
    story.append(Paragraph("LOAN & CREDIT PROFILE SUMMARY", styles['SectionHeader']))

    # Create a two-column table for profile details
    profile_table_data = [
        ['SSN:', f"XXX-XX-{str(p.get('ssn', 'N/A')).split('-')[-1]}", 'Annual Income:', p.get('annual_income_fmt', 'N/A')],
        ['Address:', p.get('address', 'N/A').replace('\n', ', '), 'Employment:', p.get('employment_status', 'N/A')],
        ['Credit Score:', str(p.get('credit_score', 'N/A')), 'Loan Requested:', p.get('loan_amount_requested_fmt', 'N/A')],
        ['Collateral Value:', p.get('collateral_value_fmt', 'N/A'), 'Monthly Alimony:', p.get('alimony_payments_monthly_fmt', 'N/A')],
    ]
    
    profile_table = Table(profile_table_data, colWidths=[1.5*inch, 2.5*inch, 1.5*inch, 2.0*inch])
    profile_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTNAME', (2, 0), (2, -1), 'Helvetica-Bold'),
        ('GRID', (0, 0), (-1, -1), 0.5, LIGHT_GREY),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(profile_table)
    story.append(Spacer(1, 0.25 * inch))
    
    # Sentiment Score Analysis
    sentiment_score = p.get('predicted_client_score', 0.0)
    sentiment_str = f"Client Sentiment Score: {sentiment_score:.2f}"
    print(sentiment_str)
    
    if sentiment_score < -0.5:
        sentiment_note = "Very Negative - Possible dissatisfaction or risk detected."
        sentiment_color = colors.red
    elif sentiment_score < 0:
        sentiment_note = "Slightly Negative - Minor follow-up recommended."
        sentiment_color = colors.orange
    else:
        sentiment_note = "Positive - No immediate concerns."
        sentiment_color = colors.darkgreen
        
    story.append(Paragraph(f"{sentiment_str} ({sentiment_note})", ParagraphStyle(name='SentimentStyle', fontName='Helvetica-Bold', fontSize=10, textColor=sentiment_color, spaceAfter=5)))
    story.append(Spacer(1, 0.35 * inch))

    # --- Build the PDF ---
    doc.build(story)


# ----------------------------------------------------------------------
## MAIN ORCHESTRATOR FUNCTION (Modified)
# ----------------------------------------------------------------------

def generate_all_client_pdfs(df_profiles_scored: pd.DataFrame, df_transactions_raw: pd.DataFrame, output_dir: str):
    """
    Generates a separate Loan Profile PDF and Bank Statement PDF for every client 
    in the scored profile data.
    
    Args:
        df_profiles_scored: The final DataFrame with client profiles and scores.
        df_transactions_raw: The raw transaction data.
        output_dir: The *directory path* where the final PDFs should be saved.
    """

    print("REPORT: Starting PDF generation for all clients...")
    
    # --- Data Cleaning and Formatting ---
    df_transactions_raw['description'] = df_transactions_raw['description'].fillna('N/A')
    df_transactions_raw['date'] = pd.to_datetime(df_transactions_raw['date']) 
    
    # FIX: Convert amount and balance to numeric float types before formatting
    df_transactions_raw['amount'] = df_transactions_raw['amount'].apply(clean_and_convert_currency)
    df_transactions_raw['balance'] = df_transactions_raw['balance'].apply(clean_and_convert_currency)

    # These lines now work because 'amount' and 'balance' are floats
    df_transactions_raw['amount_fmt'] = df_transactions_raw['amount'].apply(lambda x: f"${x:,.2f}")
    df_transactions_raw['balance_fmt'] = df_transactions_raw['balance'].apply(lambda x: f"${x:,.2f}")
    
    # Pre-format financial columns in the profiles dataframe
    currency_cols = ['annual_income', 'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly']
    for col in currency_cols:
        # We ensure the data being formatted is numeric or treated as 0 for safe formatting
        column_data = df_profiles_scored[col].apply(lambda x: clean_and_convert_currency(x) if isinstance(x, str) else x)
        df_profiles_scored[f'{col}_fmt'] = column_data.apply(lambda x: f"${x:,.0f}" if pd.notna(x) and x > 0 else "N/A")
    
    # Group the raw transactions data by client_id
    grouped_transactions = df_transactions_raw.groupby('client_id')
    
    # --- Main Execution: Iterate and Generate ---
    for index, p in df_profiles_scored.iterrows():
        client_id = p['client_id']
        client_last_name = p.get('last_name', 'NoName')
        
        # Get transactions for the current client (or an empty DataFrame) and sort by date
        if client_id in grouped_transactions.groups:
            client_transactions = grouped_transactions.get_group(client_id).sort_values(by='date')
        else:
            client_transactions = pd.DataFrame() 

        # --- Generate Loan Report PDF ---
        loan_pdf_filename = f"Loan_Profile_{client_id}_{client_last_name}.pdf"
        # We assume output_dir is a path that works, or we use the local directory if not needed.
        full_loan_path = f"{output_dir}/{loan_pdf_filename}" if output_dir else loan_pdf_filename
        
        create_client_loan_report_pdf(p, client_id, full_loan_path)
        print(f"Generated Loan Profile for Client {client_id}: {full_loan_path}")
        
        # --- Generate Bank Statement PDF ---
        statement_pdf_filename = f"Bank_Statement_{client_id}_{client_last_name}.pdf"
        full_statement_path = f"{output_dir}/{statement_pdf_filename}" if output_dir else statement_pdf_filename
        
        create_client_statement_pdf(p, client_transactions, client_id, full_statement_path)
        print(f"Generated Bank Statement for Client {client_id}: {full_statement_path}")

    print("\nAll client reports (Loan Profiles and Bank Statements) have been generated.")


# ----------------------------------------------------------------------
## MAIN EXECUTION BLOCK (Updated for the new structure)
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Ensure you have the required dataframes to test
    # NOTE: You'll need to create the 'test_code/databases' directory and the CSV files for this to run.
    try:
        df_profiles = pd.read_csv('test_code/databases/all_profiles2.csv')[:10]
        df_transactions = pd.read_csv('test_code/databases/all_statements2.csv')[:1000]
        
        # Assuming we want to output to the local directory for this example
        # In a real setup, you might pass a specific path like 'output_pdfs'
        generate_all_client_pdfs(df_profiles, df_transactions, '') 

    except FileNotFoundError as e:
        print(f"ERROR: Could not find required test file: {e}. Please ensure 'test_code/databases/all_profiles2.csv' and 'test_code/databases/all_statements2.csv' exist.")
