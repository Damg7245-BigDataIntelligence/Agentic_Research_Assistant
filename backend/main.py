from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import json
import time

# Import existing modules
from vector_storage_service import (
    store_in_pinecone,
    search_pinecone
)
from llm_service import generate_response_with_gemini  # Use only Gemini as per requirement
from agents.web_search_graph import run_web_search_workflow

# Pass the lifespan to FastAPI
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp")

# Request/Response models
class QuestionRequest(BaseModel):
    question: str
    vector_db: str
    quarter_filter: Optional[List[str]] = None
    top_k: int = 5
    document_id: Optional[str] = None  # For user-uploaded PDFs

class AvailableQuartersResponse(BaseModel):
    quarters: List[str]

# Add new endpoint model
class WebSearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 5

# API Endpoints
@app.get("/")
async def root():
    return {"message": "NVIDIA Financial RAG API is running"}

@app.get("/health")
async def health_check():
    """Check the health of connected services"""
    status = {
        "api": "healthy",
        "pinecone": "unknown",
    }
    
    # Check Pinecone
    if PINECONE_API_KEY:
        try:
            import pinecone
            indexes = pinecone.list_indexes()
            status["pinecone"] = "healthy"
            status["indexes"] = indexes
        except Exception as e:
            status["pinecone"] = f"unhealthy: {str(e)}"
    else:
        status["pinecone"] = "not configured"

    return status

@app.get("/available_quarters", response_model=AvailableQuartersResponse)
async def get_available_quarters():
    """Get all available quarters from the vector databases"""
    quarters = {"2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                    "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                    "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                    "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
                    "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"}
    
    return {"quarters": sorted(list(quarters))}

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Answer a question using RAG with the specified vector DB and quarter filter"""
    try:
        start_time = time.time()
    
        filter_dict = {}
        if request.quarter_filter and len(request.quarter_filter) > 0:
            filter_dict["quarter"] = {"$in": request.quarter_filter}
        if request.document_id:
            filter_dict["document_id"] = request.document_id
                
        # Search Pinecone
        index_name = "nvidia-financials"
        context_chunks = search_pinecone(
            request.question,
            index_name=index_name,
            filter_dict=filter_dict if filter_dict else None,
            top_k=request.top_k
        )
        
        # If no context chunks found, return an appropriate message
        if not context_chunks:
            return {
                "answer": "I couldn't find any relevant information to answer your question. Please try a different question or adjust your filters.",
                "context_chunks": [],
                "processing_time": time.time() - start_time,
                "token_info": None
            }
        
        # Extract text from context chunks
        context_text = "\n\n".join([chunk["text"] for chunk in context_chunks])
        
        # Generate answer with Gemini
        answer, token_info = generate_response_with_gemini(
            request.question,
            context_text,
            model_name="gemini-1.5-pro"  # Using Gemini as per your requirement
        )
        
        return {
            "answer": answer,
            "context_chunks": context_chunks,
            "processing_time": time.time() - start_time,
            "token_info": token_info
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")