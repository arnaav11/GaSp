

import fitz # PyMuPDF
import sys

def extract_text_from_pdf(pdf_path):
    """
    Extracts all text from a PDF file and returns it as a single string.

    Args:
        pdf_path (str): The file path to the PDF document.

    Returns:
        str: A string containing all the text extracted from the PDF.
             Returns an empty string if the file cannot be opened or processed.
    """
    try:
        
        document = fitz.open(pdf_path)
        all_text = ""

        
        for page_number in range(document.page_count):
            page = document.load_page(page_number)
            text = page.get_text()
            all_text += text + "\n" 

        
        document.close()
        
        return all_text

    except FileNotFoundError:
        print(f"Error: The file at '{pdf_path}' was not found.")
        return ""
    except Exception as e:
        print(f"An error occurred while processing the PDF: {e}")
        return ""

if __name__ == "__main__":
    
    
    if len(sys.argv) < 2:
        print("Usage: python your_dad.py <path_to_your_pdf_file>")
    else:
        pdf_to_test = sys.argv[1]
        print(f"--- Running PDF Text Extractor for '{pdf_to_test}' ---")
        
        extracted_content = extract_text_from_pdf(pdf_to_test)

        if extracted_content:
            print("\nExtracted text successfully:\n")
            print("====================================")
            print(extracted_content)
            print("====================================")
        else:
            print("\nCould not extract text. Check the error message above.")