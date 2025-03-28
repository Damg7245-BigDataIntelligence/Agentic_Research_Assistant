from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
from typing import Dict, List
from pinecone_db import AgenticResearchAssistant
from research_graph import initialize_research_graph, run_research_graph

# Define lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the research graph
    try:
        initialize_research_graph()
        print("Research graph initialized during server startup")
    except Exception as e:
        print(f"Error initializing research graph: {e}")
        raise e
    
    yield  # Server is running
    
    # Cleanup (if needed)
    print("Shutting down research graph...")

# Initialize FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Define the input model
class SearchRequest(BaseModel):
    query: str
    year_quarter_dict: Dict[str, List[str]]  # Accept string keys & string lists

# New model for LangGraph research
class ResearchRequest(BaseModel):
    query: str
    year_quarter_dict: Dict[str, List[str]]
    mode: str = "combined"  # "pinecone", "web_search", or "combined"

# API Endpoints
@app.get("/")
async def root():
    return {"message": "Nvidia Agentic Research Assistant"}

@app.get("/available_quarters", response_model=AvailableQuartersResponse)
async def get_available_quarters():
    """Get all available quarters from the vector databases"""
    quarters = {"2021-Q1", "2021-Q2", "2021-Q3", "2021-Q4",
                    "2022-Q1", "2022-Q2", "2022-Q3", "2022-Q4",
                    "2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4",
                    "2024-Q1", "2024-Q2", "2024-Q3", "2024-Q4",
                    "2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"}
    
    return {"quarters": sorted(list(quarters))}

@app.post("/summarize_using_pinecone")
def search(request: SearchRequest):
    assistant = AgenticResearchAssistant()
    response = assistant.search_pinecone_db(request.query, request.year_quarter_dict)
    return {"response": response}    

@app.post("/web_search")
async def web_search_endpoint(request: WebSearchRequest):
    """Search the web for information about NVIDIA"""
    try:
        from agents.web_search_agent import WebSearchAgent
        agent = WebSearchAgent()
        results = agent.search_news(request.query, request.num_results)
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching web: {str(e)}")

@app.post("/research")
async def research_endpoint(request: ResearchRequest):
    """
    Run the LangGraph research workflow to get information.
    """
    try:
        start_time = time.time()
        
        print(f"Research request received with mode: {request.mode}")
        print(f"Query: {request.query}")
        print(f"Year/quarter dict: {request.year_quarter_dict}")
        
        # Validate mode
        valid_modes = ["pinecone", "web_search", "snowflake", "combined"]
        if request.mode not in valid_modes:
            return {"error": f"Invalid mode '{request.mode}'. Must be one of: {', '.join(valid_modes)}"}
        
        # Validate year_quarter_dict for pinecone, snowflake, and combined modes
        if request.mode in ["pinecone", "snowflake", "combined"] and (not request.year_quarter_dict or not any(request.year_quarter_dict.values())):
            return {"error": f"For {request.mode} search, at least one year and quarter must be selected"}
        
        try:
            # Run the research workflow
            result = run_research_graph(
                query=request.query,
                year_quarter_dict=request.year_quarter_dict,
                mode=request.mode
            )
            
            # Format the response
            processing_time = time.time() - start_time
            
            return {
                "result": result,
                "processing_time": processing_time,
                "mode": request.mode
            }
        except Exception as e:
            import traceback
            print(f"ERROR in run_research_graph: {e}")
            traceback.print_exc()
            raise
            
    except Exception as e:
        import traceback
        print(f"OUTER ERROR: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error running research workflow: {str(e)}")    

@app.get("/pinecone_data_check")
async def check_pinecone_data():
    """
    Check which years and quarters have data in Pinecone and return sample records
    """
    try:
        assistant = AgenticResearchAssistant()
        
        # Get index statistics
        stats = assistant.index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        
        if total_vectors == 0:
            return {
                "status": "empty",
                "message": "No data found in Pinecone index."
            }
        
        # First, let's find what years and quarters are available
        # We'll use a minimal vector to get diverse results (not focused on semantic similarity)
        import numpy as np
        dummy_vector = np.zeros(assistant.dimension).tolist()
        
        results = assistant.index.query(
            vector=dummy_vector,
            top_k=500,  # Request a large number to get diverse samples
            include_metadata=True
        )
        
        # Extract years and quarters from metadata
        year_quarter_map = {}
        
        for match in results.get("matches", []):
            metadata = match.get("metadata", {})
            year = metadata.get("year", "unknown")
            quarter = metadata.get("quarter", "unknown")
            
            # Initialize the year entry if it doesn't exist
            if year not in year_quarter_map:
                year_quarter_map[year] = {}
            
            # Initialize the quarter entry if it doesn't exist
            if quarter not in year_quarter_map[year]:
                year_quarter_map[year][quarter] = []
            
            # Only keep up to 3 samples per quarter
            if len(year_quarter_map[year][quarter]) < 3:
                year_quarter_map[year][quarter].append({
                    "id": match.get("id", "unknown"),
                    "score": match.get("score", 0),
                    "header": metadata.get("header", "No header"),
                    "text_preview": metadata.get("text", "")[:200] + "..." if metadata.get("text") else "No text"
                })
        
        # Format the results for readability
        formatted_results = {
            "status": "success",
            "total_vectors": total_vectors,
            "years_available": sorted(list(year_quarter_map.keys())),
            "data": {}
        }
        
        # Build a structured response with year/quarter hierarchy
        for year in sorted(year_quarter_map.keys()):
            formatted_results["data"][year] = {}
            
            for quarter in sorted(year_quarter_map[year].keys()):
                formatted_results["data"][year][quarter] = year_quarter_map[year][quarter]
        
        return formatted_results
        
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error checking Pinecone data: {str(e)}"
        }    