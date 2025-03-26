from nvidia_pdf_extraction import fetch_nvidia_financial_reports
from s3_utils import fetch_s3_urls, get_presigned_url, upload_to_s3
from mistral_ocr_markdown import extract_text_from_pdf

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


if __name__ == '__main__':
    # Run the pipeline only once
    reports = fetch_pdf_s3_upload()
    convert_markdown_s3_upload()