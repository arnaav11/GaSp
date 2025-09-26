import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import StringIO
import datetime

# --- 1. Simulate Reading CSV Files ---

# Data from all_profiles2.csv (Profile Data)
profiles_csv = """client_id,first_name,last_name,ssn,address,annual_income,employment_status,credit_score,loan_amount_requested,collateral_value,alimony_payments_monthly,sentiment_score
1,Deborah,Newman,165-40-2844,"8928 Gomez Shoal\nEast Mark, NJ 32109",136817,Employed,612,13723,0,0,-0.97
2,Randy,Martinez,708-22-1813,"97604 Julia Rest\nWilkinsonhaven, MT 98490",167238,Self-Employed,838,32931,79449,0,0.04
3,Charles,Henderson,774-71-9919,"52585 Kathleen Hollow Apt. 843\nSouth Raymondview, KY 05968",142614,Self-Employed,715,36903,57272,1420,-0.46
4,Darlene,Brock,593-62-9717,"2234 Johnny Cliffs\nLake Vanessa, ME 96557",199520,Self-Employed,843,17739,45291,0,-0.42
"""
df_profiles = pd.read_csv(StringIO(profiles_csv))

# Data from initial transaction input (Statement Data Source 1)
transactions_csv_1 = """client_id,date,description,type,amount,balance
1,2025-04-06,Streaming Service - Apple App Store,DEBIT,365.46,7019.54
1,2025-04-10,"Refund from Garcia, Wang and Allen",CREDIT,464.71,7484.25
1,2025-04-11,Health Products - Pharmacy,DEBIT,202.87,7281.38
1,2025-04-12,Dining - Whole Foods,DEBIT,1.49,7279.89
1,2025-04-12,Payroll Deposit,CREDIT,4500.00,11779.89
1,2025-04-15,Mortgage Payment,DEBIT,2100.00,9679.89
1,2025-04-18,Gas Station - Shell,DEBIT,55.12,9624.77
1,2025-04-22,Online Retail - Amazon,DEBIT,123.45,9501.32
"""
df_transactions_1 = pd.read_csv(StringIO(transactions_csv_1))

# Data from all_statements2.csv (Statement Data Source 2 - NEW DATA)
transactions_csv_2 = """client_id,date,description,type,amount,balance
1,2025-04-06,Streaming Service - Apple App Store,DEBIT,365.46,7019.54
1,2025-04-10,"Refund from Garcia, Wang and Allen",CREDIT,464.71,7484.25
1,2025-04-11,Health Products - Pharmacy,DEBIT,202.87,7281.38
1,2025-04-12,Dining - Whole Foods,DEBIT,1.49,7279.89
1,2025-04-14,Prescription - CVS,DEBIT,417.37,6862.52
1,2025-04-18,Freelance Work,CREDIT,317.22,7179.74
"""
df_transactions_2 = pd.read_csv(StringIO(transactions_csv_2))


# --- 2. Combine Transaction Data ---
# Since both transaction files are for client_id 1 and have some overlapping dates/data, 
# we'll use the DISTINCT, combined set. The most reliable way is to concatenate 
# and then remove duplicates if they exist, or simply assume df_transactions_2 
# is the new/updated transaction ledger. For robustness, we will combine all unique records.
df_transactions = pd.concat([df_transactions_1, df_transactions_2]).drop_duplicates(subset=['client_id', 'date', 'amount', 'balance'], keep='last').reset_index(drop=True)


# --- 3. Data Cleaning and Formatting ---
df_transactions['description'] = df_transactions['description'].fillna('N/A')
# Convert to datetime for proper sorting
df_transactions['date'] = pd.to_datetime(df_transactions['date']) 

# Pre-format financial columns in the profiles dataframe
currency_cols = ['annual_income', 'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly']
for col in currency_cols:
    df_profiles[f'{col}_fmt'] = df_profiles[col].apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")

# Pre-format financial columns in the transactions dataframe
df_transactions['amount_fmt'] = df_transactions['amount'].apply(lambda x: f"${x:,.2f}")
df_transactions['balance_fmt'] = df_transactions['balance'].apply(lambda x: f"${x:,.2f}")


# --- 4. Setup PDF Styles ---
styles = getSampleStyleSheet()
NAVY_BLUE = colors.HexColor('#003366')
LIGHT_GREY = colors.HexColor('#F0F0F0')
WHITE = colors.white
BLACK = colors.black

styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, spaceAfter=15, alignment=0, textColor=NAVY_BLUE))
styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold', fontSize=12, spaceAfter=5, textColor=NAVY_BLUE, backColor=LIGHT_GREY))
styles.add(ParagraphStyle(name='ProfileBody', fontName='Helvetica', fontSize=10, spaceAfter=5, leading=12))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica-Oblique', fontSize=8, spaceAfter=5, textColor=colors.grey))


# --- 5. PDF Generation Function ---

def create_client_report(client_id, profile_data, transactions_data):
    """Generates a combined Loan and Statement PDF report for a single client."""
    
    # Get the single row of profile data for the client
    p = profile_data.iloc[0]
    
    pdf_filename = f"Client_Report_{client_id}_{p['last_name']}.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                            leftMargin=0.75*inch, rightMargin=0.75*inch,
                            topMargin=0.75*inch, bottomMargin=0.75*inch)
    story = []

    # --- Document Header ---
    story.append(Paragraph("Comprehensive Client Financial Report", styles['TitleStyle']))
    story.append(Paragraph(f"**Client Name:** {p['first_name']} {p['last_name']} | **Client ID:** {p['client_id']}", styles['SectionHeader']))
    story.append(Paragraph(f"**Report Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}", styles['SmallText']))
    story.append(Spacer(1, 0.25 * inch))

    # ----------------------------------------------------------------------
    ## LOAN & CREDIT PROFILE SUMMARY
    # ----------------------------------------------------------------------
    story.append(Paragraph("LOAN & CREDIT PROFILE SUMMARY", styles['SectionHeader']))

    # Create a two-column table for profile details
    profile_table_data = [
        ['SSN:', f"XXX-XX-{str(p['ssn']).split('-')[-1]}", 'Annual Income:', p['annual_income_fmt']],
        ['Address:', p['address'].replace('\n', ', '), 'Employment:', p['employment_status']],
        ['Credit Score:', str(p['credit_score']), 'Loan Requested:', p['loan_amount_requested_fmt']],
        ['Collateral Value:', p['collateral_value_fmt'], 'Monthly Alimony:', p['alimony_payments_monthly_fmt']],
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
    sentiment_str = f"Client Sentiment Score: **{p['sentiment_score']:.2f}**"
    if p['sentiment_score'] < -0.5:
        sentiment_note = "Very Negative - Possible dissatisfaction or risk detected."
        sentiment_color = colors.red
    elif p['sentiment_score'] < 0:
        sentiment_note = "Slightly Negative - Minor follow-up recommended."
        sentiment_color = colors.orange
    else:
        sentiment_note = "Positive - No immediate concerns."
        sentiment_color = colors.darkgreen
        
    story.append(Paragraph(f"{sentiment_str} ({sentiment_note})", ParagraphStyle(name='SentimentStyle', fontName='Helvetica-Bold', fontSize=10, textColor=sentiment_color, spaceAfter=5)))
    story.append(Spacer(1, 0.35 * inch))

    # ----------------------------------------------------------------------
    ## RECENT TRANSACTION HISTORY
    # ----------------------------------------------------------------------
    story.append(Paragraph("RECENT TRANSACTION HISTORY", styles['SectionHeader']))
    
    if not transactions_data.empty:
        # Prepare transaction data for table (converting date back to string for display)
        transaction_data_list = transactions_data[['date', 'description', 'type', 'amount_fmt', 'balance_fmt']].copy()
        transaction_data_list['date'] = transaction_data_list['date'].dt.strftime('%Y-%m-%d')
        transaction_data_list = transaction_data_list.values.tolist()
        
        table_header = ['Date', 'Description', 'Type', 'Amount', 'Balance']
        
        # Determine the opening/closing balance for the statement period
        opening_balance = transactions_data['balance'].iloc[0] - transactions_data['amount'].iloc[0] if transactions_data['type'].iloc[0] == 'CREDIT' else transactions_data['balance'].iloc[0] + transactions_data['amount'].iloc[0]
        closing_balance = transactions_data['balance_fmt'].iloc[-1]
        
        story.append(Paragraph(f"**Opening Balance:** ${opening_balance:,.2f} | **Closing Balance:** {closing_balance}", styles['ProfileBody']))
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
    print(f"Generated report for Client {client_id}: {pdf_filename}")


# --- 6. Main Execution: Iterate and Generate ---
# Group the combined transactions data by client_id
grouped_transactions = df_transactions.groupby('client_id')

# Iterate over each client in the profile data
for index, profile_row in df_profiles.iterrows():
    client_id = profile_row['client_id']
    
    # Get transactions for the current client (or an empty DataFrame if none exist) and sort by date
    if client_id in grouped_transactions.groups:
        # Sort by date before passing to the generator
        client_transactions = grouped_transactions.get_group(client_id).sort_values(by='date')
    else:
        client_transactions = pd.DataFrame() # Empty DataFrame

    # Generate the report for the client
    create_client_report(client_id, 
                         df_profiles[df_profiles['client_id'] == client_id], 
                         client_transactions)

print("\nAll client reports have been generated.")