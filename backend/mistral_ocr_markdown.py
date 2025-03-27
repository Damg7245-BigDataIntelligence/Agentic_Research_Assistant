import os
from mistralai import Mistral
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
INPUT_FILE_PATH = "pdf/2025/2025_Third_Quarter.pdf"
OUTPUT_FILE_PATH = "markdown/2025/2025_Third_Quarter.md"

# Initialize Mistral client
mistral_client = Mistral(api_key=MISTRAL_API_KEY)

def extract_text_from_pdf(pdf_url):
    """Extract text from PDF using Mistral OCR API."""
    try:
        # Process the document using Mistral OCR
        ocr_response = mistral_client.ocr.process(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url,
            },
        )

        # Combine all pages into one markdown document
        all_pages_markdown = []

        for page in ocr_response.pages:
            all_pages_markdown.append(page.markdown)

        markdown_content = "\n\n".join(all_pages_markdown)

        print(f"Successfully extracted {len(markdown_content)} characters with Mistral OCR")
        
        return markdown_content

    except Exception as e:
        raise Exception(f"Failed to extract text using Mistral OCR: {str(e)}")


# def main():
#     try:
#         # Step 1: Generate presigned URL for the input PDF file in S3
#         pdf_url = get_presigned_url(INPUT_FILE_PATH)
        
#         # Step 2: Extract text and convert to Markdown using Mistral OCR API
#         markdown_content = extract_text_from_pdf(pdf_url)

#         # Step 3: Upload Markdown content to S3 bucket
#         upload_to_s3(OUTPUT_FILE_PATH, markdown_content)

#     except Exception as e:
#         print(f"An error occurred: {str(e)}")

# if __name__ == "__main__":
#     main()
