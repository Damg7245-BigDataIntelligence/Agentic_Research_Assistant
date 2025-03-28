import os
import logging
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from markdown_chunking import chunk_markdown_by_headers
import requests
from urllib.parse import urlparse

load_dotenv()

def extract_filename_year_quarter(url: str):
    """
    Extracts the filename, year, and quarter from a given URL.
    
    Args:
        url (str): The URL containing the file path.
    
    Returns:
        dict: A dictionary containing 'filename', 'year', and 'quarter'.
    """
    # Extract path from URL before query parameters
    parsed_url = urlparse(url)
    path = parsed_url.path
    
    # Get the filename from the path
    filename = os.path.basename(path)
    
    # Extract year and quarter using the provided function logic
    def extract_year_and_quarter(file_path: str):
        base = os.path.basename(file_path)
        name, _ = os.path.splitext(base)
        parts = name.split("_")
        if len(parts) >= 2:
            year = parts[0]
            quarter_word = parts[1].lower()
            mapping = {"first": "1", "second": "2", "third": "3", "fourth": "4"}
            quarter = mapping.get(quarter_word, "Unknown")
            return year, quarter
        return "Unknown", "Unknown"
    year, quarter = extract_year_and_quarter(filename)
    return filename, year, quarter

class AgenticResearchAssistant:
    def __init__(self):
        # Configure Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        
        # Load environment variables
        self.PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        print(f"PINECONE_API_KEY: {self.PINECONE_API_KEY}")
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.PINECONE_API_KEY)
        self.index_name = "nvidia-agentic-research-assistant"
        self.dimension = 384  # Matching the embedding model's output size
        
        # Configure Gemini API
        genai.configure(api_key=self.GOOGLE_API_KEY)
        print(f"GOOGLE_API_KEY: {self.GOOGLE_API_KEY}")
        self.gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        
        # Check and create Pinecone index if it doesn’t exist
        if self.index_name not in [index["name"] for index in self.pc.list_indexes()]:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logging.info(f"Index '{self.index_name}' created.")
        else:
            logging.info(f"Index '{self.index_name}' already exists.")
        
        # Connect to the index and print its stats
        self.index = self.pc.Index(self.index_name)
        logging.info(f"Pinecone index stats: {self.index.describe_index_stats()}")
        
        # Load Sentence Transformer Model
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logging.info("Sentence Transformer model loaded.")

    def process_markdown(self, file_path):
        """Reads a markdown file and processes it into chunks."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                markdown_text = f.read()
            chunks = chunk_markdown_by_headers(markdown_text)
            logging.info(f"Extracted {len(chunks)} chunks from markdown file.")
            return chunks
        except Exception as e:
            logging.error(f"Error reading markdown file: {e}")
            return []

    def insert_embeddings(self, presigned_url, year, quarter, filename):
        """Processes markdown from a presigned URL, generates embeddings, and inserts into Pinecone."""
        try:
            # Fetch markdown content from the presigned URL
            response = requests.get(presigned_url)
            response.raise_for_status()  # Raise an error for failed requests
            markdown_text = response.text
            
            # Process chunks from markdown content
            chunks = chunk_markdown_by_headers(markdown_text)
            if not chunks:
                logging.warning("No chunks extracted. Skipping embedding.")
                return

            # Extract only the text content from chunks
            chunk_texts = [chunk["content"] for chunk in chunks]

            # Generate embeddings
            embeddings = self.model.encode(chunk_texts).tolist()
            logging.info(f"Generated embeddings for {len(embeddings)} chunks.")

            # Prepare batch upserts for Pinecone
            pinecone_data = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                metadata = {
                    "text": chunk["content"],
                    "header": chunk.get("header", "No Header"),
                    "level": str(chunk.get("level", "Unknown")),
                    "part": str(chunk.get("part")) if chunk.get("part") is not None else "None",
                    "year": year,
                    "quarter": quarter,
                    "filename": filename
                }
                pinecone_data.append((f"{year}_{quarter}_{i}", embedding, metadata))
            
            # Insert data into Pinecone in batch
            self.index.upsert(pinecone_data)
            logging.info(f"Inserted {len(pinecone_data)} chunks into Pinecone successfully.")
        except Exception as e:
            logging.error(f"Error processing presigned URL: {e}")
        
    def search_pinecone_db(self, query, year_quarter_dict, top_k=20):
        """Search for relevant chunks in Pinecone, filtering by multiple years and quarters, and generate a response using Gemini."""
        query_embedding = self.model.encode([query]).tolist()
        try:
            # Construct metadata filter for multiple years and quarters
            filter_criteria = {
                "$or": [
                    {"year": {"$eq": str(year)}, "quarter": {"$in": [str(q) for q in quarters]}}
                    for year, quarters in year_quarter_dict.items()
                ]
            }
            print(filter_criteria)

            # Perform a filtered search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_criteria  # Apply filtering
            )

            matches = results.get("matches", [])
            if not matches:
                logging.warning(f"No relevant matches found for the given year-quarter combinations.")
                return "No relevant information found for the specified year and quarters."

            # Extract matched texts along with their metadata
            retrieved_data = [(match["metadata"]["text"], match["metadata"]["year"], match["metadata"]["quarter"]) for match in matches]
            print("retrieved_data: ", retrieved_data)
            
            # Create context for Gemini
            context = "\n".join([f"Year: {year}, Quarter: {quarter} - {text}" for text, year, quarter in retrieved_data])
            prompt = f"""You are an AI assistant tasked with analyzing Nvidia's financial data. 
                    Below is relevant financial information retrieved from a vector database, with each entry associated with a specific year and quarter. 
                    Use this context to answer the question accurately.
                    Question: {query}
                    Context: {context}
                    """

            # Generate response using Gemini
            response = self.gemini_model.generate_content(prompt)
            print("pineceone output: ", response.text)
            return response.text
        except Exception as e:
            logging.error(f"Error during search: {e}")
            return "Error occurred during search."
        
