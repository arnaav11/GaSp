import streamlit as st
import pandas as pd
import numpy as np
import time 
from main import run_gasp_pipeline

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
        background-color: #b19cd9;
        height: 1px;
        border: none;
    }
    
    /* =================================================================== */
    /* === 2. TYPOGRAPHY === */
    /* =================================================================== */
    /* All Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #b19cd9; /* Light Purple Accent */
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
        color: #b19cd9;
    }

    /* Info/Success/Warning/Error boxes */
    [data-testid="stInfo"], [data-testid="stSuccess"], [data-testid="stWarning"], [data-testid="stError"] {
        background-color: #121212;
        border: 1px solid #b19cd9;
        border-radius: 10px;
        color: #FFFFFF;
    }
    /* Set icon color to purple for info boxes */
    [data-testid="stInfo"] .st-emotion-cache-1wivap2 {
        color: #b19cd9;
    }

    /* Metric Widget */
    [data-testid="stMetric"] {
        background-color: #121212;
        border: 1px solid #2a2a2a;
        padding: 15px;
        border-radius: 10px;
    }
    [data-testid="stMetricLabel"] {
        color: #b19cd9; /* Purple label for metrics */
    }
    [data-testid="stMetricValue"], [data-testid="stMetricDelta"] {
        color: #FFFFFF; /* White value and delta */
    }

    /* Number Input Widget */
    input[type="number"] {
        background-color: #1e1e1e;
        color: #FFFFFF;
        border: 1px solid #b19cd9;
        border-radius: 5px;
    }
    
    /* =================================================================== */
    /* === 4. BUTTONS & FILE UPLOADER === */
    /* =================================================================== */

    /* Primary Buttons (in main content) */
    .stButton > button {
        background-color: #b19cd9;
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
        border: 1px solid #b19cd9;
        background-color: transparent;
        color: #e0e0e0;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: rgba(177, 156, 217, 0.1);
        color: #b19cd9;
        border-color: #c0b2d9;
    }
    [data-testid="stSidebar"] .stButton button:focus {
        background-color: rgba(177, 156, 217, 0.2); /* Highlight selected button */
        color: #FFFFFF;
    }

    /* === THIS IS THE CORRECTED CODE FOR THE FILE UPLOADER === */

    /* Main container for the file uploader (the "drag and drop" area) */
    [data-testid="stFileUploader"] {
        border-radius: 10px;
        border: 2px dashed #b19cd9;      /* Purple dashed border */
        background-color: #000000;       /* Black background */
        padding: 1rem;
    }

    /* The 'Browse files' button inside the uploader */
    section[data-testid="stFileUploader"] button {
        background-color: #000000 !important; /* Black background */
        color: #000000 !important;            /* White text */
        border: 1px solid #b19cd9 !important; /* Purple border */
    }

    /* Text inside the uploader (e.g., 'Drag and drop file here') */
    [data-testid="stFileUploader"] span, [data-testid="stFileUploader"] p {
        color: #000000 !important; /* White text */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Initialize session state for page navigation ---
if 'selected_section' not in st.session_state:
    st.session_state.selected_section = "Client Documents"
if 'assessment_initiated' not in st.session_state:
    st.session_state.assessment_initiated = False
if 'pipeline_output' not in st.session_state:
    st.session_state.pipeline_output = None # Store the result dictionary here


# --- Functions for navigation ---
def go_to_section(section_name):
    st.session_state.selected_section = section_name

def start_assessment():
    st.session_state.assessment_initiated = True
    # Immediately trigger the analysis when the button is pressed
    all_uploaded_files = []
    file_keys = ['personal_docs', 'financial_docs', 'asset_docs', 'additional_docs']
    
    # Collect all file objects
    for key in file_keys:
        files = st.session_state.get(key)
        # We pass the file objects (which contain the .name property) to the pipeline
        if files and isinstance(files, list):
            all_uploaded_files.extend(files)

    # Store the output of the pipeline in session state
    st.session_state.pipeline_output = run_gasp_pipeline(all_uploaded_files)
    
    go_to_section("Assessment Results")


# --- Sidebar and Navigation ---
with st.sidebar:
    st.markdown(
        """
        <div style="background-color: #b19cd9; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 20px;">
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
    # This button uses the primary style defined in the CSS
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
    st.markdown("_Securely upload the required documents for a thorough analysis. Our platform supports various file types and ensures data privacy and security. **Note:** Only the file path/name is used for simulation._")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("📂 Personal & Tax Forms")
        st.file_uploader(
            "Upload client personal details, SSN, and tax forms (PDF, JPG, PNG)",
            type=["pdf", "jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="personal_docs"
        )
        st.subheader("🏦 Bank Statements & Financial Records")
        st.file_uploader(
            "Upload bank statements and financial records (PDF, CSV)",
            type=["pdf", "csv"],
            accept_multiple_files=True,
            key="financial_docs"
        )
    with col2:
        st.subheader("💼 Asset & Debt Documentation")
        st.file_uploader(
            "Upload proof of assets, collateral, and debt history (PDF, JPG, PNG)",
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
        st.number_input(
            "Estimated Credit Score",
            min_value=300,
            max_value=850,
            step=1,
            help="A value between 300 and 850.",
            key="credit_score"
        )
        st.number_input(
            "Annual Salary ($)",
            min_value=0,
            step=1000,
            key="annual_salary"
        )
    with col4:
        st.subheader("💰 Financial Standing")
        st.number_input(
            "Total Debts ($)",
            min_value=0,
            step=1000,
            key="total_debts"
        )
        st.number_input(
            "Total Assets ($)",
            min_value=0,
            step=1000,
            key="total_assets"
        )
    with col5:
        st.subheader("📈 Loan Details")
        st.number_input(
            "Investment/Loan Price ($)",
            min_value=0,
            step=1000,
            key="investment_price"
        )
        st.number_input(
            "Interest Rate (%)",
            min_value=0.0,
            max_value=100.0,
            step=0.1,
            format="%.2f",
            key="interest_rate"
        )

elif st.session_state.selected_section == "AI Model Details":
    st.header("3. Key Data Points for AI Model")
    st.markdown("_Our AI model is a proprietary, multi-faceted engine trained on a variety of crucial data points to provide its comprehensive assessment. The model considers the following factors:_")
    st.markdown("---")
    st.markdown(
        """
        - <b><span style='color:#b19cd9;'>Credit History</span></b>: _Detailed analysis of payment history, credit utilization, and credit age._
        - <b><span style='color:#b19cd9;'>Income & Employment</span></b>: _Verification of income stability, employment history, and salary trends._
        - <b><span style='color:#b19cd9;'>Debt & Assets</span></b>: _A full evaluation of the debt-to-income ratio and asset-to-debt ratio._
        - <b><span style='color:#b19cd9;'>Personal Information</span></b>: _Demographic and residential data for contextual analysis._
        - <b><span style='color:#b19cd9;'>Loan Details</span></b>: _Specifics of existing and proposed loans, including interest rates and terms._
        - <b><span style='color:#b19cd9;'>Transactional Data & Digital Footprint</span></b>: _Secure analysis of banking and online activity for behavioral patterns._
        - <b><span style='color:#b19cd9;'>Sentiment Analysis from documents</i></b>: _AI-driven text analysis to gauge qualitative risk factors from provided documents._
        """,
        unsafe_allow_html=True
    )

elif st.session_state.selected_section == "Assessment Results":
    st.header("4. Assessment Results")
    st.markdown("_A comprehensive, AI-driven report of the client's financial profile. This includes risk scores, key insights, and actionable recommendations._")
    st.markdown("---")

    if st.session_state.assessment_initiated and st.session_state.pipeline_output:
        # Get the actual, calculated data
        results = st.session_state.pipeline_output
        
        st.success("Assessment complete! A detailed report has been generated below.")

        # Display the pipeline result to confirm the files were processed
        st.subheader("Pipeline Status")
        st.code("\n".join(results['pipeline_summary']))
        st.markdown("---") 

        # --- DISPLAY RESULTS USING EXTRACTED/CALCULATED VALUES ---
        
        st.subheader("Overall Risk Score")
        # Using columns to create a visually appealing metric layout
        cols = st.columns(3)
        
        # Display the extracted credit score (612)
        cols[0].metric(label="Credit Risk Score", value=str(results['credit_score']), delta=None if results['credit_score'] == 750 else f"{results['credit_score'] - 750}")
        
        # Display the calculated Fraud and Viability based on the credit score
        cols[1].metric(label="Fraud Risk", value=results['fraud'])
        cols[2].metric(label="Investment Viability", value=results['viability'])

        st.markdown("---")
        st.subheader("Key Financial Metrics")
        
        col_metrics1, col_metrics2 = st.columns(2, gap="large")
        with col_metrics1:
            # Display the calculated DTI based on the credit score
            st.metric(label="Debt-to-Income Ratio", value=results['dti'], delta="-2%", delta_color="inverse", help="Lower is better.")
            # Display the dynamically extracted Annual Salary
            st.metric(label="Annual Salary (Extracted)", value=f"${results['annual_salary']:,.0f}", help="Extracted from uploaded documents.") 
        with col_metrics2:
            # Display the calculated Approval rate based on the credit score
            st.metric(label="Projected Loan Approval", value=results['approval'], delta="Adjusted based on Credit Score") 
            st.metric(label="Investment ROI (1-year)", value="15%", delta="+2% from forecast") # Still placeholder

        st.markdown("---")
        st.subheader("Visual Analysis")
        st.write("_A visual breakdown of the client's financial health, demonstrating key trends and areas of risk._")
        # Placeholder image with a theme-matching URL
        st.image("https://placehold.co/1000x500/000000/b19cd9?text=Portfolio+Breakdown+Chart", use_column_width=True)

        st.markdown("---")
        st.subheader("AI-Generated Insights & Recommendations")
        # Display the calculated insight based on the credit score
        st.info(results['insights'])
    else:
        st.info("Please click the 'Start Comprehensive Assessment' button in the sidebar to begin the analysis and view the results.")
