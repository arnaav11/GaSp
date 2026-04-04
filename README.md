# GA$P: Global AI-powered Strategic Portfolio

GA$P is a Streamlit web application for **AI-assisted portfolio and loan assessment**. Users upload client documents; the backend extracts structured data from PDFs, runs a rules-based risk analysis (credit, debt-to-income, sentiment, and related factors), generates a chart of monthly credits versus debits when transaction data exists, and displays scores, metrics, and narrative insights in the UI.

## Features

- **Document upload**: Four upload areas for personal and tax-related files, bank and financial records, asset and debt documentation, and additional supporting documents (mixed file types accepted by the UI).
- **Assessment pipeline**: Saves uploads under `temp_uploaded_files/`, parses compatible PDFs, computes risk outputs, and saves a financial summary chart alongside the uploads.
- **Results dashboard**: Pipeline status log, credit risk score, fraud and investment viability labels, debt-to-income ratio, salary and debit summaries, projected loan approval outcome, a bar chart of monthly credits versus debits, and AI-style recommendation text.
- **Navigation**: Sidebar sections for Client Documents, Manual Data Entry, AI Model Details (describes factors the design targets), and Assessment Results.
- **Theming**: Custom black, white, and purple styling with Libre Baskerville typography.

## Architecture

| Component | Role |
|-----------|------|
| `main.py` | Streamlit UI, session state, file collection, and call to `run_gasp_pipeline`. |
| `test_code/pipeline.py` | PDF text extraction (PyMuPDF), analysis steps, matplotlib/seaborn chart generation. |

The pipeline is implemented as three logical steps: receive and parse files, analyze the primary client row, generate and save a chart PNG.

## Requirements

- Python 3.9 or newer is recommended (adjust if your environment differs).
- Dependencies used by the app and pipeline:

  - `streamlit`
  - `pandas`
  - `PyMuPDF` (imported as `fitz`)
  - `matplotlib`
  - `seaborn`

Install them with pip, for example:

```bash
pip install streamlit pandas PyMuPDF matplotlib seaborn
```

## Running the application

From the repository root:

```bash
streamlit run main.py
```

Streamlit prints a local URL (typically `http://localhost:8501`). Open it in your browser.

## How to use

1. Go to **Client Documents** and upload at least one file the pipeline can parse (see **Expected PDF content** below).
2. Click **Start Comprehensive Assessment** in the sidebar. The app writes files to `temp_uploaded_files/` and runs the pipeline; on success it switches to **Assessment Results**.
3. Review the pipeline log, metrics, chart, and insights on the results page.

**Manual Data Entry** and **AI Model Details** are informational in the current codebase: the manual numeric fields are not passed into `run_gasp_pipeline`, so assessment outputs are driven by extracted PDF data only.

## Expected PDF content

The extractor looks for PDF text matching internal templates:

- **Loan profile**: A section titled `LOAN & CREDIT PROFILE SUMMARY` with fields such as Client ID, Client Name, SSN, Address, Annual Income, Employment, Credit Score, Loan Requested, Collateral Value, Monthly Alimony, and Client Sentiment Score (parsed via regular expressions).
- **Transactions**: A `TRANSACTION HISTORY` block with lines like `YYYY-MM-DD` description `CREDIT` or `DEBIT` with dollar amounts and balance.

If no loan profile block is found in any uploaded PDF, the pipeline returns an error asking you to check file format and content. Files that are not PDFs or that PyMuPDF cannot open are skipped during extraction.

## Output artifacts

- Uploaded files and generated charts are stored under `temp_uploaded_files/` (created if missing). Chart filenames follow the pattern `financial_summary_<client_id>.png` when transaction data is present.

## Limitations

- Assessment logic is **heuristic** (thresholds on credit score, DTI, income, sentiment, etc.), not a trained ML model served from this repository.
- Upload widgets accept several extensions (for example PDF, CSV, images, DOCX), but the implemented parser is built around **PDF text extraction** for the structures above.
- **Manual Data Entry** values are not merged into the pipeline in `main.py`.

## Project layout (high level)

```
GaSp/
  main.py                 # Streamlit entrypoint
  test_code/
    pipeline.py           # Extraction, analysis, chart generation
    ...                   # Other modules and sample data under test_code/
```

## License

No license file is included in this repository; add one if you distribute or reuse the code.
