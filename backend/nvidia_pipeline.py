from backend.nvidia_pdf_extraction import fetch_nvidia_financial_reports
from backend.s3_utils import fetch_s3_urls, get_presigned_url, upload_to_s3
from backend.mistral_ocr_markdown import extract_text_from_pdf
from backend.pinecone_db import extract_filename_year_quarter, AgenticResearchAssistant
import time

def fetch_pdf_s3_upload():
    # Step 1: Fetch NVIDIA financial reports
    print("Step 1: Fetching financial reports...")
    reports = fetch_nvidia_financial_reports()
    print("Reports fetched successfully:")
    for report in reports:
        print(f"Fetched: {report['pdf_filename']} (Size: {report['content']} bytes)")
    return reports

def convert_markdown_s3_upload():
    # Instantiate the OCR extractor only once
    s3_urls = fetch_s3_urls("pdf/")
    s3_urls = s3_urls[1:]
    for input_url in s3_urls:
        output_url = "markdown"+input_url[3:-3]+"md"
        pdf_url = get_presigned_url(input_url)
        markdown_content = extract_text_from_pdf(pdf_url)
        upload_to_s3(output_url, markdown_content)
        print(f"{input_url} converted to md")
        time.sleep(10)

def generate_pinecone_embeddings(assistant):
    """Fetch all markdown URLs under the 'markdown' folder and convert them to presigned URLs."""
    print("Fetching markdown files...")
    markdown_urls = fetch_s3_urls("markdown/")
    markdown_urls = markdown_urls[1:]
    presigned_urls = [get_presigned_url(url) for url in markdown_urls]
    print(f"Fetched {len(presigned_urls)} markdown files.")

    # Step 2: Process each markdown file and insert embeddings into Pinecone
    for url in presigned_urls:
        print(url)
        filename, year, quarter = extract_filename_year_quarter(url)  # Extract metadata from filename
        assistant.insert_embeddings(url, year, quarter, filename)
        print(f"Inserted Embeddings for the {year} and {quarter}")


if __name__ == '__main__':
    # Run the pipeline only once
    reports = fetch_pdf_s3_upload()
    convert_markdown_s3_upload()
    assistant = AgenticResearchAssistant()
    generate_pinecone_embeddings(assistant)