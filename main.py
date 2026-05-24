"""
FastAPI Backend for AI Legal Assistant
Simplified wrapper around LangGraph workflow
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from langchain_core.messages import HumanMessage
import logging
from datetime import datetime


# Import the compiled workflow
from graph import new_workflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FastAPI Application Setup
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

app = FastAPI(
    title="AI Legal Assistant API",
    description="FastAPI wrapper for LangGraph legal assistant workflow",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Request/Response Models
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class LegalQueryRequest(BaseModel):
    """Request model for legal queries"""
    query: str = Field(..., min_length=1, description="Your legal query or question")
    
    class Config:
        example = {
            "query": "Can you generate a timeline for Arvind Kejriwal CBI case with little bit summary?"
        }


class LegalQueryResponse(BaseModel):
    """Response model for legal queries"""
    status: str = Field(default="success")
    query: str
    response: str
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(default="healthy")
    message: str
    timestamp: str


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# API Endpoints
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": "AI Legal Assistant API",
        "version": "1.0.0",
        "endpoints": {
            "POST /query": "Process legal query",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    logger.info("Health check requested")
    return HealthResponse(
        status="healthy",
        message="AI Legal Assistant API is running",
        timestamp=datetime.now().isoformat()
    )


@app.post("/query", response_model=LegalQueryResponse, tags=["Legal Assistant"])
async def process_query(request: LegalQueryRequest):
    """
    Process a legal query through the LangGraph workflow
    
    The query is passed through the compiled workflow which includes:
    - React Agent for legal analysis
    - Vector store retrieval for case law
    - Web search for recent information
    - Wikipedia search for context
    - Document drafting capabilities
    """
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        
        # Invoke the workflow with the user query
        workflow_input = {
            "messages": [HumanMessage(content=request.query)],
            "retriever_docs": [],
            "tools_used": [],
            "draft": "",
            "draft_type": ""
        }
        
        # Execute the workflow
        result = new_workflow.invoke(workflow_input)
        
        # Extract the response
        if result.get('messages'):
            response_content = result['messages'][-1].content
        else:
            response_content = "No response generated"
        
        logger.info("Query processed successfully")
        
        return LegalQueryResponse(
            status="success",
            query=request.query,
            response=response_content,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


@app.post("/chat", tags=["Legal Assistant"])
async def chat(request: LegalQueryRequest):
    """
    Alternative endpoint for chat interface
    Passes through to /query endpoint
    """
    try:
        logger.info(f"Chat request: {request.query[:100]}...")
        
        workflow_input = {
            "messages": [HumanMessage(content=request.query)],
            "retriever_docs": [],
            "tools_used": [],
            "draft": "",
            "draft_type": ""
        }
        
        result = new_workflow.invoke(workflow_input)
        
        if result.get('messages'):
            response_content = result['messages'][-1].content
        else:
            response_content = "No response generated"
        
        return {
            "status": "success",
            "query": request.query,
            "response": response_content,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}"
        )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Startup and Shutdown Events
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("=" * 60)
    logger.info("AI Legal Assistant API - Starting")
    logger.info("=" * 60)
    logger.info("✓ FastAPI initialized")
    logger.info("✓ LangGraph workflow loaded")
    logger.info("✓ Ready to process legal queries")
    logger.info("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("AI Legal Assistant API - Shutting down")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Run Application
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app=app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
