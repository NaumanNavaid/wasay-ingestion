"""
🚀 COMPLETE SINGLE-FILE CHATBOT API
No other files needed! Just this one + .env with API keys
"""

import os
from datetime import datetime
from functools import lru_cache
from typing import Literal
from uuid import uuid4

from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import AsyncQdrantClient, models

# =============================================================================
# CONFIGURATION
# =============================================================================


class Settings(BaseSettings):
    """Application settings from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # OpenAI Configuration
    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"

    # Qdrant Configuration
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "textbook_chunks"

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class QueryRequest(BaseModel):
    """Request model for asking a question."""

    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class SourceReference(BaseModel):
    """Reference to a source used in the answer."""

    source_url: str
    chapter_section: str | None = None


class QueryResponse(BaseModel):
    """Response model for a query."""

    query_id: str
    question: str
    answer: str
    sources: list[SourceReference]
    retrieved_chunk_count: int
    timestamp: str


class ContextQueryRequest(BaseModel):
    """Request model for asking about selected text."""

    selected_text: str = Field(..., min_length=10, max_length=10000)
    question: str = Field(..., min_length=1, max_length=500)


class ContextQueryResponse(BaseModel):
    """Response model for context query."""

    query_id: str
    question: str
    answer: str
    selected_text_preview: str
    timestamp: str


class ErrorResponse(BaseModel):
    """Error response model."""

    error: str
    message: str
    details: dict | None = None


# =============================================================================
# CLIENTS
# =============================================================================

# OpenAI clients
openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
chat_client = AsyncOpenAI(api_key=settings.openai_api_key)

# Qdrant client (use sync client for compatibility)
from qdrant_client import QdrantClient
qdrant_client = QdrantClient(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
)


# =============================================================================
# SERVICES
# =============================================================================


async def generate_embedding(text: str) -> list[float]:
    """Generate embedding for text using OpenAI."""
    response = await openai_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=text,
    )
    return response.data[0].embedding


def search_qdrant(query_embedding: list[float], limit: int = 5):
    """Search for similar chunks in Qdrant."""
    results = qdrant_client.search(
        collection_name=settings.qdrant_collection,
        query_vector=query_embedding,
        limit=limit,
        score_threshold=0.5,
    )
    return results


async def answer_question(question: str, top_k: int = 5) -> dict:
    """Answer a question using RAG (Retrieval-Augmented Generation)."""
    try:
        # Step 1: Generate embedding for question
        question_embedding = await generate_embedding(question)

        # Step 2: Search Qdrant for relevant chunks (sync call in async function)
        search_results = search_qdrant(question_embedding, limit=top_k)

        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the textbook to answer your question. Try rephrasing it.",
                "sources": [],
            }

        # Step 3: Build context from retrieved chunks
        context_parts = []
        sources = []

        for result in search_results:
            text = result.payload.get("text", "")
            source_url = result.payload.get("source_url", "")
            chapter = result.payload.get("chapter_section", "")

            context_parts.append(f"[{chapter or 'Unknown Section'}] (Relevance: {result.score:.2f})\n{text}")
            sources.append({
                "source_url": source_url,
                "chapter_section": chapter,
            })

        context = "\n\n".join(context_parts)

        # Step 4: Generate answer using GPT
        system_prompt = """You are a helpful AI assistant answering questions about a robotics and physical AI textbook. Use ONLY the provided context from the textbook to answer the user's question accurately.

Guidelines:
- Base your answer ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Be concise but thorough
- Use specific details from the context when available"""

        user_prompt = f"""Context from the textbook:
{context}

Question: {question}

Please provide a clear, accurate answer based on the textbook content above."""

        response = await chat_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "sources": sources,
        }

    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "sources": [],
        }


async def answer_about_selected_text(selected_text: str, question: str) -> dict:
    """Answer a question about selected text from the book."""
    try:
        system_prompt = """You are a helpful AI assistant answering questions about a robotics and physical AI textbook. The user will provide a selected passage from the book and ask you a question about it.

Guidelines:
- Answer ONLY based on the provided selected text
- If the text doesn't contain enough information to answer, say so clearly
- Be concise but thorough
- Use specific details from the text when available"""

        user_prompt = f"""Selected text from the textbook:
{selected_text}

Question: {question}

Please provide a clear, accurate answer based on the selected text above."""

        response = await chat_client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1000,
        )

        answer = response.choices[0].message.content

        return {
            "answer": answer,
            "selected_text_preview": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
        }

    except Exception as e:
        return {
            "answer": f"Error processing your question: {str(e)}",
            "selected_text_preview": selected_text[:200] + "..." if len(selected_text) > 200 else selected_text,
        }


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="Textbook RAG Chatbot",
    description="AI-powered chatbot for robotics textbook using RAG",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS
# =============================================================================


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Textbook RAG Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "ask_question": "POST /chatbot/query",
            "ask_selection": "POST /chatbot/ask-selection",
            "api_docs": "/docs",
            "health": "/health",
        },
        "features": [
            "RAG-based question answering",
            "Text selection Q&A",
            "Chapter-specific queries",
            "Source references",
        ],
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post(
    "/chatbot/query",
    response_model=QueryResponse,
    tags=["Chatbot"],
    summary="Ask a question about the textbook",
    description="Uses RAG to search the textbook and generate an answer with source references.",
)
async def query_chatbot(request: QueryRequest) -> QueryResponse:
    """Ask a question about the textbook using RAG.

    This endpoint will:
    1. Generate an embedding for your question
    2. Search for relevant textbook chunks in Qdrant
    3. Generate an answer using GPT with the retrieved context
    4. Return the answer with source references
    """
    query_id = str(uuid4())

    result = await answer_question(request.question, request.top_k)

    sources_list = [
        SourceReference(
            source_url=s["source_url"],
            chapter_section=s.get("chapter_section"),
        )
        for s in result["sources"]
    ]

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=result["answer"],
        sources=sources_list,
        retrieved_chunk_count=len(result["sources"]),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/chatbot/ask-selection",
    response_model=ContextQueryResponse,
    tags=["Chatbot"],
    summary="Ask about selected text",
    description="Select a passage from the book and ask a question about it.",
)
async def ask_about_selection(request: ContextQueryRequest) -> ContextQueryResponse:
    """Ask a question about selected text from the textbook.

    This endpoint allows you to select/highlight a passage from the book
    and ask a specific question about that text. The AI will answer
    using only the selected text as context.
    """
    query_id = str(uuid4())

    result = await answer_about_selected_text(request.selected_text, request.question)

    return ContextQueryResponse(
        query_id=query_id,
        question=request.question,
        answer=result["answer"],
        selected_text_preview=result["selected_text_preview"],
        timestamp=datetime.utcnow().isoformat(),
    )


# =============================================================================
# ERROR HANDLERS
# =============================================================================


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return ErrorResponse(
        error="http_error",
        message=exc.detail,
        details={"status_code": exc.status_code},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    return ErrorResponse(
        error="internal_error",
        message=str(exc),
        details={"type": type(exc).__name__}
    )


# =============================================================================
# MAIN - RUN SERVER
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("=" * 60)
    print("  TEXTBOOK RAG CHATBOT API STARTING")
    print("=" * 60)
    print("  API Docs:       http://localhost:8000/docs")
    print("  Query Endpoint: POST /chatbot/query")
    print("  Selection:      POST /chatbot/ask-selection")
    print("=" * 60)

    port = int(os.getenv("PORT", settings.api_port))
    uvicorn.run(
        app,
        host=settings.api_host,
        port=port,
        reload=False,
        log_level="info",
    )
