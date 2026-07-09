# api/main.py — FastAPI Server for LexTech RAG System
# This turns our RAG pipeline into a proper API
# that any application can call

import sys
import os
import time
from typing import Optional

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path so we can import our modules
# This is needed because main.py is inside the api/ folder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our RAG pipeline from Phase 4
from rag_pipeline import retrieve_chunks, build_prompt, generate_answer

# Import ChromaDB and embedding model
import chromadb
from sentence_transformers import SentenceTransformer


# ─────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────

# Create FastAPI app
# title and description show up in auto-generated docs
app = FastAPI(
    title="LexTech Support Knowledge Copilot",
    description="RAG-powered support assistant for LexTech documentation",
    version="1.0.0"
)

# CORS = Cross Origin Resource Sharing
# This allows web browsers to call our API
# Without this, browsers would block the requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # allow all origins
    allow_methods=["*"],    # allow all HTTP methods
    allow_headers=["*"],    # allow all headers
)

# Load embedding model once when server starts
# We load it once here so every request doesn't reload it
print("⏳ Loading embedding model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model ready!")

# Connect to ChromaDB
chroma_client = chromadb.PersistentClient(path="embeddings/chroma_db")
collection = chroma_client.get_collection("lextech_knowledge_base")
print(f"✅ Connected to ChromaDB ({collection.count()} chunks)")


# ─────────────────────────────────────────────────
# REQUEST/RESPONSE MODELS
# ─────────────────────────────────────────────────

# Pydantic models define the shape of our data
# FastAPI uses these to validate incoming requests
# and format outgoing responses

class AskRequest(BaseModel):
    """What the user sends when asking a question"""
    question: str           # the question text
    n_results: Optional[int] = 3  # how many chunks to retrieve

class SourceChunk(BaseModel):
    """A source chunk returned with the answer"""
    source: str
    heading: str
    score: float
    preview: str            # first 200 characters of chunk

class AskResponse(BaseModel):
    """What we send back to the user"""
    question: str
    answer: str
    sources: list[SourceChunk]
    chunks_searched: int
    response_time_seconds: float

class SearchResponse(BaseModel):
    """Response for search-only requests"""
    query: str
    results: list[SourceChunk]
    total_chunks_in_db: int


# ─────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────

# ENDPOINT 1: Health Check
# GET / → checks if server is running
@app.get("/")
def health_check():
    """
    Simple health check endpoint.
    Returns server status and database info.

    Daily life analogy:
    Like calling a restaurant to ask "are you open?"
    before driving there.
    """
    return {
        "status": "healthy",
        "service": "LexTech Support Knowledge Copilot",
        "version": "1.0.0",
        "chunks_in_database": collection.count(),
        "model": "all-MiniLM-L6-v2",
        "llm": "llama3.2"
    }


# ENDPOINT 2: Ask a Question
# POST /ask → main RAG endpoint
@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest):
    """
    Main RAG endpoint — ask any question about LexTech.

    Receives a question, retrieves relevant chunks,
    generates an answer using Llama, returns answer + sources.

    Daily life analogy:
    The main order counter at our restaurant.
    You place your order (question) and get your food (answer).
    """

    # Validate the question
    if not request.question.strip():
        # HTTPException sends an error response back
        # 400 = Bad Request (user sent invalid data)
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    if len(request.question) > 500:
        raise HTTPException(
            status_code=400,
            detail="Question too long (max 500 characters)"
        )

    # Start timing how long the request takes
    start_time = time.time()

    try:
        # Step 1: Retrieve relevant chunks
        chunks = retrieve_chunks(
            request.question,
            n_results=request.n_results
        )

        # Step 2: Build prompt
        prompt = build_prompt(request.question, chunks)

        # Step 3: Generate answer
        answer = generate_answer(prompt)

        # Calculate response time
        response_time = time.time() - start_time

        # Format source chunks for response
        sources = []
        for chunk in chunks:
            sources.append(SourceChunk(
                source=chunk["source"],
                heading=chunk["heading"],
                score=round(chunk["score"], 4),
                preview=chunk["content"][:200]
            ))

        # Return the complete response
        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            chunks_searched=len(chunks),
            response_time_seconds=round(response_time, 2)
        )

    except Exception as e:
        # If anything goes wrong, return a 500 error
        # 500 = Internal Server Error
        raise HTTPException(
            status_code=500,
            detail=f"Error generating answer: {str(e)}"
        )


# ENDPOINT 3: Search Only (no answer generation)
# GET /search?q=your+question
@app.get("/search", response_model=SearchResponse)
def search_chunks(q: str, n: int = 3):
    """
    Search the knowledge base without generating an answer.
    Returns the most relevant chunks for a query.

    Useful for:
    - Debugging retrieval
    - Seeing what documents exist
    - Building custom frontends

    Daily life analogy:
    Like asking the librarian "where would I find books about X?"
    without actually getting a book yet.
    """

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty"
        )

    chunks = retrieve_chunks(q, n_results=n)

    results = []
    for chunk in chunks:
        results.append(SourceChunk(
            source=chunk["source"],
            heading=chunk["heading"],
            score=round(chunk["score"], 4),
            preview=chunk["content"][:200]
        ))

    return SearchResponse(
        query=q,
        results=results,
        total_chunks_in_db=collection.count()
    )


# ENDPOINT 4: List All Chunks
# GET /chunks
@app.get("/chunks")
def list_chunks():
    """
    Returns all chunks stored in the database.
    Useful for inspecting what's been indexed.
    """

    # Get all chunks from ChromaDB
    all_chunks = collection.get()

    chunks_list = []
    for i in range(len(all_chunks["ids"])):
        chunks_list.append({
            "id": all_chunks["ids"][i],
            "source": all_chunks["metadatas"][i]["source"],
            "heading": all_chunks["metadatas"][i]["heading"],
            "word_count": all_chunks["metadatas"][i]["word_count"],
            "preview": all_chunks["documents"][i][:100]
        })

    return {
        "total_chunks": len(chunks_list),
        "chunks": chunks_list
    }