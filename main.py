import streamlit as st
import pandas as pd
import os
from test_code.pipeline import run_gasp_pipeline

# --- Page Configuration ---
st.set_page_config(
    page_title="GA$P: AI-Powered Portfolio Assessment",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Inject custom CSS for a consistent Black, White, and Purple theme ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&display=swap');
    
    /* =================================================================== */
    /* === 1. ROOT & MAIN STYLES === */
    /* =================================================================== */
    html, body, [class*="st-"] {
        font-family: 'Libre Baskerville', serif;
    }

    /* Main App Background */
    .stApp {
        background-color: #000000; /* Pure Black Background */
        color: #FFFFFF; /* White text by default */
    }
    
    /* Top Header Bar */
    [data-testid="stHeader"] {
        background-color: #000000;
        border-bottom: 1px solid #2a2a2a;
    }

    /* Sidebar Background */
    [data-testid="stSidebar"] {
        background-color: #121212; /* Slightly off-black for contrast */
        border-right: 1px solid #2a2a2a;
    }
    /* === SIDEBAR TOGGLE BUTTON HIDE (<< / >>) === */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    /* Horizontal Divider Line */
    hr {
        background-color: #917cb9;
        height: 1px;
        border: none;
    }
    
    /* =================================================================== */
    /* === 2. TYPOGRAPHY === */
    /* =================================================================== */
    /* All Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #917cb9; /* Light Purple Accent */
        font-style: normal;
    }

    /* All standard text, labels, etc. */
    p, label, .st-emotion-cache-16txtl3, .st-emotion-cache-1y4p8pa, [data-testid="stMarkdownContainer"] {
        color: #FFFFFF !important;
    }

    /* Sidebar markdown text */
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] em {
        color: #FFFFFF !important;
    }

    /* =================================================================== */
    /* === 3. CUSTOM CONTAINERS & WIDGETS === */
    /* =================================================================== */

    /* Main header container */
    .main-header-container {
        text-align: center;
        background: #121212;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 2rem;
        border: 1px solid #2a2a2a;
        box-shadow: 0 4px 12px rgba(177, 156, 217, 0.1);
    }
    .main-header-container h1 {
        color: #917cb9;
    }

    /* Info/Success/Warning/Error boxes */
    [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] {
        background-color: #121212;
        border: 1px solid #917cb9;
        border-radius: 10px;
        color: #FFFFFF;
    }
    /* Set icon color to purple for info boxes */
    [data-testid="stInfo"] .st-emotion-cache-1wivap2 {
        color: #917cb9;
    }

    /* Metric Widget */
    [data-testid="stMetric"] {
        background-color: #121212;
        border: 1px solid #2a2a2a;
        padding: 15px;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"] {
        color: #917cb9; /* Purple label for metrics */
    }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: #FFFFFF; /* White value and delta */
    }

    /* Number Input Widget */
    input[type="number"] {
        background-color: #1e1e1e;
        color: #FFFFFF;
        border: 1px solid #917cb9;
        border-radius: 5px;
    }
    
    /* =================================================================== */
    /* === 4. BUTTONS & FILE UPLOADER === */
    /* =================================================================== */

    /* Primary Buttons (in main content) */
    .stButton > button {
        background-color: #917cb9;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .stButton > button:hover {
        background-color: #c0b2d9;
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.4);
    }

    /* Sidebar Navigation Buttons */
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        margin-top: 10px;
        border: 1px solid #917cb9;
        background-color: transparent;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(177, 156, 217, 0.1);
        color: #917cb9;
        border-color: #c0b2d9;
    }
    [data-testid="stSidebar"] .stButton button:focus {
        background-color: rgba(177, 156, 217, 0.2); /* Highlight selected button */
        color: #FFFFFF;
    }

    /* Main container for the file uploader (the "drag and drop" area) */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 2px dashed #917cb9;     /* Purple dashed border */
        background-color: #121212;      /* Dark background for the box */
        padding: 1rem;
    }

    /* The 'Browse files' button inside the uploader */
    [data-testid="stFileUploader"] button {
        background-color: #917cb9 !important; /* Purple background */
        color: #FFFFFF !important;            /* White text */
        border: 1px solid #917cb9 !important; 
    }

    /* Text inside the uploader (e.g., 'Drag and drop file here') */
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {
        color: #FFFFFF !important; /* White text */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Initialize session state for page navigation ---
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "Client Documents"
if 'assessment_initiated' not in st.session_state:
    st.session_state.assessment_initiated = False
if 'pipeline_output' not in st.session_state:
    st.session_state.pipeline_output = None # Store the result dictionary here


# --- Functions for navigation and logic ---
def go_to_section(section_name):
    st.session_state.selected_section = section_name

def start_assessment():
    st.session_state.assessment_initiated = True
    
    # Create a temporary directory for uploaded files if it doesn't exist
    temp_dir = "temp_uploaded_files"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    all_uploaded_files = []
    file_keys = ['personal_docs', 'financial_docs', 'asset_docs', 'additional_docs']
    
    # Collect all UploadedFile objects from session state
    for key in file_keys:
        files = st.session_state.get(key)
        if files and isinstance(files, list):
            all_uploaded_files.extend(files)

    # Check if any files were uploaded
    if not all_uploaded_files:
        st.warning("Please upload at least one document before starting the assessment.")
        st.session_state.assessment_initiated = False
        return

    # Save uploaded files to the temporary directory and collect their paths
    all_file_paths = []
    for uploaded_file in all_uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        all_file_paths.append(file_path)

    # Run the backend pipeline with the file paths
    try:
        with st.spinner('Running AI-powered assessment... This may take a moment.'):
            st.session_state.pipeline_output = run_gasp_pipeline(all_file_paths)
    except Exception as e:
        st.error(f"An error occurred during assessment: {e}")
        st.session_state.pipeline_output = None
        st.session_state.assessment_initiated = False
        return
    
    go_to_section("Assessment Results")


# --- Sidebar and Navigation ---
with st.sidebar:
    st.markdown(
        """
        <div style="background-color: #917cb9; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
            <h2 style="color: black; margin: 0; font-family: 'Libre Baskerville', serif;">GA$P Navigation</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown("_Navigate through the application by clicking the sections below._")

    if st.button("Client Documents", key="btn_docs"):
        go_to_section("Client Documents")
    if st.button("Manual Data Entry", key="btn_manual"):
        go_to_section("Manual Data Entry")
    if st.button("AI Model Details", key="btn_details"):
        go_to_section("AI Model Details")
    if st.button("Assessment Results", key="btn_results"):
        go_to_section("Assessment Results")

    st.markdown("---")
    st.markdown("_Ready to assess? Click the button below!_")
    st.button("Start Comprehensive Assessment 🚀", key="btn_assess", on_click=start_assessment, use_container_width=True)

# --- Main Content ---
st.markdown(
    "<div class='main-header-container'><h1>GA$P: AI-Powered Portfolio Assessment</h1></div>",
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center; margin-bottom: 2rem;'>
        <p style='font-size: 1.2em; font-style: normal;'>
        Welcome to GA$P, the <b>Global AI-powered Strategic Portfolio</b> assessment platform. This tool is designed to provide a comprehensive, data-driven evaluation of a client's financial health, leveraging advanced machine learning models to analyze diverse data points.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
st.markdown("---")

# --- Content for Each Section ---
if st.session_state.selected_section == "Client Documents":
    st.header("1. Client Portfolio & Document Submission")
    st.markdown("_Securely upload the required documents for a thorough analysis. Our platform supports various file types and ensures data privacy and security._")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("📂 Personal & Tax Forms")
        st.file_uploader(
            "Upload client personal details and tax forms (e.g., Loan_Profile.pdf)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="personal_docs"
        )
        st.subheader("🏦 Bank Statements & Financial Records")
        st.file_uploader(
            "Upload bank statements and financial records (e.g., Bank_Statement.pdf)",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            key="financial_docs"
        )
    with col2:
        st.subheader("💼 Asset & Debt Documentation")
        st.file_uploader(
            "Upload proof of assets, collateral, and debt history",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="asset_docs"
        )
        st.subheader("📄 Additional Supporting Documents")
        st.file_uploader(
            "Upload any other relevant loan or alimony history documents",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="additional_docs"
        )

elif st.session_state.selected_section == "Manual Data Entry":
    st.header("2. Manual Data Points")
    st.markdown("_Please enter specific financial data manually to augment the AI model's automated assessment. This helps ensure higher accuracy._")

    col3, col4, col5 = st.columns(3, gap="large")
    with col3:
        st.subheader("📊 Credit & Income")
        st.number_input("Estimated Credit Score", min_value=300, max_value=850, step=1, help="A value between 300 and 850.", key="credit_score")
        st.number_input("Annual Salary ($)", min_value=0, step=1000, key="annual_salary_manual")
    with col4:
        st.subheader("💰 Financial Standing")
        st.number_input("Total Debts ($)", min_value=0, step=1000, key="total_debts")
        st.number_input("Total Assets ($)", min_value=0, step=1000, key="total_assets")
    with col5:
        st.subheader("📈 Loan Details")
        st.number_input("Investment/Loan Price ($)", min_value=0, step=1000, key="investment_price")
        st.number_input("Interest Rate (%)", min_value=0.0, max_value=100.0, step=0.1, format="%.2f", key="interest_rate")

elif st.session_state.selected_section == "AI Model Details":
    st.header("3. Key Data Points for AI Model")
    st.markdown("_Our AI model is a proprietary, multi-faceted engine trained on a variety of crucial data points to provide its comprehensive assessment. The model considers the following factors:_")
    st.markdown("---")
    st.markdown(
        """
        - <b><span style='color:#917cb9;'>Credit History</span></b>: _Detailed analysis of payment history, credit utilization, and credit age._
        - <b><span style='color:#917cb9;'>Income & Employment</span></b>: _Verification of income stability, employment history, and salary trends._
        - <b><span style='color:#917cb9;'>Debt & Assets</span></b>: _A full evaluation of the debt-to-income ratio and asset-to-debt ratio._
        - <b><span style='color:#917cb9;'>Personal Information</span></b>: _Demographic and residential data for contextual analysis._
        - <b><span style='color:#917cb9;'>Loan Details</span></b>: _Specifics of existing and proposed loans, including interest rates and terms._
        - <b><span style='color:#917cb9;'>Transactional Data & Digital Footprint</span></b>: _Secure analysis of banking and online activity for behavioral patterns._
        - <b><span style='color:#917cb9;'>Sentiment Analysis from documents</i></b>: _AI-driven text analysis to gauge qualitative risk factors from provided documents._
        """,
        unsafe_allow_html=True
    )

elif st.session_state.selected_section == "Assessment Results":
    st.header("4. Assessment Results")
    st.markdown("_A comprehensive, AI-driven report of the client's financial profile. This includes risk scores, key insights, and actionable recommendations._")
    st.markdown("---")

    if st.session_state.assessment_initiated and st.session_state.pipeline_output:
        results = st.session_state.pipeline_output
        
        if "error" in results:
            st.error(f"Assessment failed: {results['error']}")
        elif isinstance(results, dict):
            st.success("Assessment complete! A detailed report has been generated below.")

            st.subheader("Pipeline Status")
            if 'pipeline_summary' in results and isinstance(results['pipeline_summary'], list):
                st.code("\n".join(results['pipeline_summary']))
            else:
                st.code("Pipeline execution successful, but no status summary was returned.")
                
            st.markdown("---") 

            st.subheader("Overall Risk Score")
            cols = st.columns(3)
            cols[0].metric(label="Credit Risk Score", value=str(results.get('credit_score', 'N/A')))
            cols[1].metric(label="Fraud Risk", value=results.get('fraud', 'N/A'))
            cols[2].metric(label="Investment Viability", value=results.get('viability', 'N/A'))

            st.markdown("---")
            st.subheader("Key Financial Metrics")
            
            col_metrics1, col_metrics2 = st.columns(2, gap="large")
            with col_metrics1:
                st.metric(label="Debt-to-Income Ratio", value=results.get('dti', 'N/A'), delta="-2%", delta_color="inverse", help="Lower is better.")
                salary_value = results.get('annual_salary', 0)
                st.metric(label="Annual Salary (Extracted)", value=f"${salary_value:,.0f}" if isinstance(salary_value, (int, float)) else str(salary_value), help="Extracted from uploaded documents.") 
            with col_metrics2:
                debit_value = results.get('total_debit', 0)
                st.metric(label="Total Debit (from Statement)", value=f"${debit_value:,.2f}" if isinstance(debit_value, (int, float)) else str(debit_value), help="Total debits calculated from transaction data.") 
                st.metric(label="Projected Loan Approval", value=results.get('approval', 'N/A'), delta="Adjusted based on Credit Score") 

            st.markdown("---")
            st.subheader("Visual Analysis")
            st.write("_A visual breakdown of the client's financial health, demonstrating key trends and areas of risk._")
            st.image("https://placehold.co/1000x500/000000/917cb9?text=Portfolio+Breakdown+Chart", use_container_width=True)

            st.markdown("---")
            st.subheader("AI-Generated Insights & Recommendations")
            st.info(results.get('insights', 'No insights generated.'))
        else:
            st.error("Assessment failed: The pipeline did not return a valid result dictionary.")

    else:
        st.info("Please upload your documents and click the 'Start Comprehensive Assessment' button in the sidebar to begin.")
