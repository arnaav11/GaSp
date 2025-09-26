import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import datetime

# --- 1. PDF Setup ---
styles = getSampleStyleSheet()
NAVY_BLUE = colors.HexColor('#003366')
LIGHT_GREY = colors.HexColor('#F0F0F0')
WHITE = colors.white
BLACK = colors.black

styles.add(ParagraphStyle(name='TitleStyle', fontName='Helvetica-Bold', fontSize=18, spaceAfter=15, alignment=0, textColor=NAVY_BLUE))
styles.add(ParagraphStyle(name='SectionHeader', fontName='Helvetica-Bold', fontSize=12, spaceAfter=5, textColor=NAVY_BLUE, backColor=LIGHT_GREY))
styles.add(ParagraphStyle(name='ProfileBody', fontName='Helvetica', fontSize=10, spaceAfter=5, leading=12))
styles.add(ParagraphStyle(name='SmallText', fontName='Helvetica-Oblique', fontSize=8, spaceAfter=5, textColor=colors.grey))


# --- 2. Report Generation Function (Called by Main Orchestrator) ---

# FIX: Function signature matches the call from main.py:
# create_client_report(analysis_results, df_trans, FINAL_OUTPUT_PATH)
def create_client_report(df_profiles_scored: pd.DataFrame, df_transactions_raw: pd.DataFrame, output_path: str):
    """
    Generates a combined Loan and Statement PDF report for every client 
    in the scored profile data.
    
    Args:
        df_profiles_scored: The final DataFrame from gauri_clienvalidity.py 
                            (includes scores, assessment, and client info).
        df_transactions_raw: The raw transaction data (for the transaction history table).
        output_path: The path where the final PDF should be saved (the path itself 
                     is usually just used for saving the *file path* in this design, 
                     but we'll output PDFs).
    """

    print("REPORT: Starting PDF generation for all clients...")
    
    # --- Data Cleaning and Formatting (Moved inside the function) ---
    df_transactions_raw['description'] = df_transactions_raw['description'].fillna('N/A')
    df_transactions_raw['date'] = pd.to_datetime(df_transactions_raw['date']) 
    df_transactions_raw['amount_fmt'] = df_transactions_raw['amount'].apply(lambda x: f"${x:,.2f}")
    df_transactions_raw['balance_fmt'] = df_transactions_raw['balance'].apply(lambda x: f"${x:,.2f}")
    
    # Pre-format financial columns in the profiles dataframe
    currency_cols = ['annual_income', 'loan_amount_requested', 'collateral_value', 'alimony_payments_monthly']
    for col in currency_cols:
         # Use .get(col, 0) to safely handle columns that might be missing from the final scored DF
        df_profiles_scored[f'{col}_fmt'] = df_profiles_scored.get(col, 0).apply(lambda x: f"${x:,.0f}" if x > 0 else "N/A")
    
    # Group the raw transactions data by client_id
    grouped_transactions = df_transactions_raw.groupby('client_id')
    
    # --- Main Execution: Iterate and Generate ---
    for index, p in df_profiles_scored.iterrows():
        client_id = p['client_id']
        
        # Get transactions for the current client (or an empty DataFrame if none exist) and sort by date
        if client_id in grouped_transactions.groups:
            client_transactions = grouped_transactions.get_group(client_id).sort_values(by='date')
        else:
            client_transactions = pd.DataFrame() 

        # --- Report Generation Logic (Internal Function) ---
        
        # FIX: Filename should use client_id and last_name for clarity
        pdf_filename = f"Client_Report_{client_id}_{p.get('last_name', 'NoName')}.pdf"
        
        # The output_path argument in main.py is likely intended to be the directory path, 
        # but we use the locally constructed filename for saving the PDF.
        doc = SimpleDocTemplate(pdf_filename, pagesize=letter,
                                leftMargin=0.75*inch, rightMargin=0.75*inch,
                                topMargin=0.75*inch, bottomMargin=0.75*inch)
        story = []

        # --- Document Header ---
        story.append(Paragraph("Comprehensive Client Financial Report", styles['TitleStyle']))
        story.append(Paragraph(f"**Client Name:** {p.get('first_name', 'N/A')} {p.get('last_name', 'N/A')} | **Client ID:** {client_id}", styles['SectionHeader']))
        story.append(Paragraph(f"**Report Date:** {datetime.datetime.now().strftime('%Y-%m-%d')}", styles['SmallText']))
        
        # Display the final predictive assessment
        assessment_style = ParagraphStyle(name='Assessment', fontName='Helvetica-Bold', fontSize=11, spaceAfter=15, textColor=NAVY_BLUE)
        story.append(Paragraph(f"**Final Assessment:** {p.get('assessment', 'N/A')}", assessment_style))

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
        sentiment_score = p.get('sentiment_score', 0.0)
        sentiment_str = f"Client Sentiment Score: **{sentiment_score:.2f}**"
        
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

        # ----------------------------------------------------------------------
        ## RECENT TRANSACTION HISTORY
        # ----------------------------------------------------------------------
        story.append(Paragraph("RECENT TRANSACTION HISTORY", styles['SectionHeader']))
        
        if not client_transactions.empty:
            # Prepare transaction data for table (converting date back to string for display)
            transaction_data_list = client_transactions[['date', 'description', 'type', 'amount_fmt', 'balance_fmt']].copy()
            transaction_data_list['date'] = transaction_data_list['date'].dt.strftime('%Y-%m-%d')
            transaction_data_list = transaction_data_list.values.tolist()
            
            table_header = ['Date', 'Description', 'Type', 'Amount', 'Balance']
            
            # Determine the opening/closing balance for the statement period
            try:
                # Assuming transactions are sorted oldest to newest (as done above)
                first_transaction = client_transactions.iloc[0]
                opening_balance = first_transaction['balance'] - first_transaction['amount'] if first_transaction['type'] == 'CREDIT' else first_transaction['balance'] + first_transaction['amount']
                closing_balance = client_transactions['balance_fmt'].iloc[-1]
            except Exception:
                opening_balance = 0.00
                closing_balance = "$0.00"
            
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

    print("\nAll client reports have been generated.")
    # NOTE: The function returns nothing, as it generates files instead of a DataFrame.
    # We rely on the printed messages to confirm success.