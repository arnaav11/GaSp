import streamlit as st
from text_extract import extract_text_from_pdf

# --- Page Configuration ---
st.set_page_config(
    page_title="GASP: AI-Powered Portfolio Assessment",
    page_icon="💸",
    
    layout="wide"
)

# --- Header Section ---
st.title("GASP: AI-Powered Portfolio Assessment")
st.markdown(
    """
    <p style='font-size: 18px;'>
    Welcome to GASP. Upload your client's financial and personal documents for a comprehensive AI-driven assessment. Our model analyzes various data points to provide a thorough evaluation of their portfolio, finances, and credit standing.
    </p>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --- Document Upload Section ---
st.header("1. Client Portfolio & Document Submission")
st.subheader("Upload the required documents below.")

col1, col2 = st.columns(2)

with col1:
    st.info("Personal Details & Tax Forms")
    personal_docs = st.file_uploader(
        "Upload client personal details, SSN, and tax forms (PDF, JPG)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.info("Bank Statements & Financial Records")
    financial_docs = st.file_uploader(
        "Upload bank statements and financial records (PDF, CSV)",
        type=["pdf"],
        accept_multiple_files=True
    )

with col2:
    st.info("Asset & Debt Documentation")
    asset_docs = st.file_uploader(
        "Upload proof of assets, collateral, and debt history (PDF, JPG)",
        type=["pdf"],
        accept_multiple_files=True
    )

    st.info("Additional Supporting Documents")
    additional_docs = st.file_uploader(
        "Upload any other relevant loan or alimony history documents",
        type=["pdf"],
        accept_multiple_files=True
    )

# --- Manual Data Entry Section ---
st.markdown("---")
st.header("2. Manual Data Points")
st.subheader("Please enter specific financial data manually for a more accurate assessment.")

col3, col4, col5 = st.columns(3)

with col3:
    credit_score = st.number_input(
        "Estimated Credit Score",
        min_value=300,
        max_value=850,
        step=1,
        help="A value between 300 and 850."
    )
    
    annual_salary = st.number_input(
        "Annual Salary ($)",
        min_value=0,
        step=1000
    )

with col4:
    total_debts = st.number_input(
        "Total Debts ($)",
        min_value=0,
        step=1000
    )

    total_assets = st.number_input(
        "Total Assets ($)",
        min_value=0,
        step=1000
    )

with col5:
    investment_price = st.number_input(
        "Investment/Loan Price ($)",
        min_value=0,
        step=1000
    )

    interest_rate = st.number_input(
        "Interest Rate (%)",
        min_value=0.0,
        max_value=100.0,
        step=0.1,
        format="%.2f"
    )

# --- Important Data Points for Training Section ---
# st.markdown("---")
# st.header("3. Key Data Points for AI Model")
# st.markdown(
#     """
#     <p>
#     Our AI model is trained on a variety of crucial data points to provide its assessment, including:
#     </p>
#     """,
#     unsafe_allow_html=True
# )

# st.text("— Credit History")
# st.text("— Income & Employment")
# st.text("— Debt & Assets")
# st.text("— Personal Information")
# st.text("— Loan Details")
# st.text("— Transactional Data & Digital Footprint")
# st.text("— Sentiment Analysis from documents")

filepaths = []

for i in personal_docs:
    filepath = f'pdfs/{i.name}'
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(i.getvalue())

    filepaths.append(filepath)

for i in asset_docs:
    filepath = f'pdfs/{i.name}'
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(i.getvalue())

    filepaths.append(filepath)

for i in financial_docs:
    filepath = f'pdfs/{i.name}'
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(i.getvalue())

    filepaths.append(filepath)

for i in additional_docs:
    filepath = f'pdfs/{i.name}'
    with open(filepath, 'wb') as pdf_file:
        pdf_file.write(i.getvalue())

    filepaths.append(filepath)



# --- Submission Button ---
st.markdown("---")
if st.button("Start Assessment", help="Click to begin the AI assessment."):
    texts = []
    for i in filepaths:
        texts.append(extract_text_from_pdf(i))
        st.success(texts)
    # st.success("Assessment initiated! Since this is a UI-only demo, no backend processing will occur.")
    st.balloons()