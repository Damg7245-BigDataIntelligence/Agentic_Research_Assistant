import os
import boto3
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Fetch the credentials and region from the environment variables
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
bucket_name = os.getenv('AWS_S3_BUCKET_NAME')

print(aws_region, bucket_name)

# Initialize a session using AWS credentials
s3_client = boto3.client(
    's3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key
)

# Function to upload binary content (e.g., PDF content) directly to S3
def upload_file_to_s3(file_content, filname, folder=None):
    """
    Uploads binary content (e.g., PDF) directly to S3.

    :param file_content: Binary content of the file.
    :param s3_key: Name of the file in S3.
    :param folder: Optional folder name in the S3 bucket (default is None).
    :return: True if upload is successful, False otherwise.
    """
    try:
        # If a folder is specified, prepend it to the key
        s3_key = f"{folder}/{filname}"

        # Upload the binary content to S3
        s3_client.put_object(Bucket=bucket_name, Key=s3_key, Body=file_content)
        print(f"File uploaded successfully to {bucket_name}/{s3_key}")
        return f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_key}"
    except Exception as e:
        print(f"Error uploading binary content: {e}")
        return False
    
def upload_pdf_to_s3(file_content: bytes, original_filename: str, document_id: str) -> str:
    """
    Uploads PDF to S3.
    Returns URL for the uploaded file.
    """
    try:
        # Upload original PDF
        pdf_key = f"documents/pdf/{document_id}/{original_filename}"
        s3_client.put_object(
            Bucket=bucket_name,
            Key=pdf_key,
            Body=file_content,
            ContentType='application/pdf'
        )
        return f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{pdf_key}"
    except Exception as e:
        raise Exception(f"Failed to upload PDF to S3: {e}")

def fetch_s3_urls(base_path):
    """
    Fetches all file URLs from a specific folder in the S3 bucket.

    :param base_path: The base folder path in the S3 bucket (e.g., "pdf/").
    :return: A list of full S3 URLs for files within the specified folder.
    """
    try:
        # Ensure the base_path ends with a slash (S3 treats folders as prefixes)
        if not base_path.endswith('/'):
            base_path += '/'

        # List objects with the specified prefix (base_path)
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=base_path)

        # Check if 'Contents' exists in the response (it won't if no files are found)
        if 'Contents' not in response:
            print(f"No files found in folder: {base_path}")
            return []

        # Construct full S3 URLs for each file
        s3_urls = []
        for obj in response['Contents']:
            key = obj['Key']  # The file key (e.g., "pdf/2021/Q1.pdf")
            s3_url = f"{key}"

            s3_urls.append(s3_url)

        print(f"Found {len(s3_urls)} files in folder: {base_path}")
        return s3_urls

    except Exception as e:
        print(f"Error fetching file paths: {e}")
        return []
    
def get_presigned_url(key):
    """Generate a presigned URL for the PDF file in S3."""
    presigned_url = s3_client.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket_name, 'Key': key},
        ExpiresIn=3600  # URL valid for 1 hour
    )
    print("Pre-signed url: ",presigned_url)
    return presigned_url

def upload_to_s3(key, content):
    """Upload the Markdown file to S3."""
    s3_client.put_object(Bucket=bucket_name, Key=key, Body=content, ContentType="text/markdown")
    print(f"Markdown file uploaded successfully to s3://{bucket_name}/{key}")