from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv
import httpx
import groq
import os
import uuid
import json

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is required.")
if not SUPABASE_URL:
    raise RuntimeError("SUPABASE_URL is required.")
if not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_KEY is required.")

groq_client = groq.Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "text-embedding-3-small"  # via OpenAI-compatible Groq or use a free alternative

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
MAX_HISTORY = 4

# Supabase REST headers
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

# -----------------------------
# FASTAPI INIT
# -----------------------------
app = FastAPI(title="DocMind API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# DATA MODELS
# -----------------------------
class QuestionRequest(BaseModel):
    session_id: str
    question: str

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    source_chunks: list[str]
    session_id: str

# -----------------------------
# IN-MEMORY SESSION STORE
# Stores only chat history (lightweight — not FAISS indexes)
# Documents are persisted in Supabase
# -----------------------------
session_history: dict = {}

def get_history(session_id: str) -> list:
    return session_history.get(session_id, [])

def save_history(session_id: str, question: str, answer: str):
    if session_id not in session_history:
        session_history[session_id] = []
    session_history[session_id].append({
        "question": question,
        "answer": answer
    })
    # Keep only last MAX_HISTORY turns
    session_history[session_id] = session_history[session_id][-MAX_HISTORY:]

# -----------------------------
# EMBEDDING VIA GROQ
# Groq supports OpenAI-compatible embeddings
# -----------------------------
async def get_embedding(text: str) -> list[float]:
    """Get embedding using Groq's embedding endpoint."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",  # Groq embedding model
                "input": text[:8000],  # Groq embedding input limit
            }
        )

        if response.status_code != 200:
            # Fallback: use a simple hash-based pseudo embedding for now
            # Replace with a real embedding API if Groq embeddings unavailable
            raise HTTPException(
                status_code=500,
                detail=f"Embedding API error: {response.text}"
            )

        data = response.json()
        return data["data"][0]["embedding"]

async def get_embedding_batch(texts: list[str]) -> list[list[float]]:
    """Get embeddings for multiple texts."""
    embeddings = []
    for text in texts:
        emb = await get_embedding(text)
        embeddings.append(emb)
    return embeddings

# -----------------------------
# SUPABASE OPERATIONS
# -----------------------------
async def store_chunks(session_id: str, chunks: list[str], sources: list[str], embeddings: list[list[float]]):
    """Store document chunks and embeddings in Supabase."""
    rows = [
        {
            "session_id": session_id,
            "content": chunk,
            "source": source,
            "embedding": embedding,
        }
        for chunk, source, embedding in zip(chunks, sources, embeddings)
    ]

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/documents",
            headers=SUPABASE_HEADERS,
            json=rows,
        )

        if response.status_code not in (200, 201):
            raise HTTPException(
                status_code=500,
                detail=f"Supabase insert error: {response.text}"
            )

async def search_chunks(session_id: str, query_embedding: list[float], top_k: int = TOP_K):
    """Search for similar chunks using pgvector cosine similarity."""
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/match_documents",
            headers=SUPABASE_HEADERS,
            json={
                "query_embedding": query_embedding,
                "match_session_id": session_id,
                "match_count": top_k,
            }
        )

        if response.status_code != 200:
            raise HTTPException(
                status_code=500,
                detail=f"Supabase search error: {response.text}"
            )

        return response.json()

async def delete_session_chunks(session_id: str):
    """Delete all chunks for a session (cleanup)."""
    async with httpx.AsyncClient(timeout=30) as client:
        await client.delete(
            f"{SUPABASE_URL}/rest/v1/documents?session_id=eq.{session_id}",
            headers=SUPABASE_HEADERS,
        )

async def session_has_documents(session_id: str) -> bool:
    """Check if session has any uploaded documents."""
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/documents?session_id=eq.{session_id}&limit=1",
            headers={**SUPABASE_HEADERS, "Prefer": "count=exact"},
        )
        content_range = response.headers.get("content-range", "0")
        try:
            total = int(content_range.split("/")[-1])
            return total > 0
        except Exception:
            return len(response.json()) > 0

# -----------------------------
# TEXT PROCESSING
# -----------------------------
def chunk_text(text: str) -> list[str]:
    """Sentence-aware chunking with overlap."""
    # Split on sentence boundaries first
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= CHUNK_SIZE:
            current_chunk += " " + sentence
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            # Start new chunk with overlap from previous
            overlap_start = max(0, len(current_chunk) - CHUNK_OVERLAP)
            current_chunk = current_chunk[overlap_start:] + " " + sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return [c for c in chunks if len(c) > 50]  # Filter out tiny fragments

# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(context: str, history: list, question: str) -> str:
    history_text = ""
    for turn in history[-MAX_HISTORY:]:
        history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

    return f"""You are DocMind, a precise document analysis assistant.

STRICT RULES:
- Answer ONLY from the provided context chunks
- If the context does not contain enough information, say: "I couldn't find this information in the document."
- Cite sources by mentioning the filename when relevant
- Be concise and clear
- Never hallucinate or use outside knowledge

CONVERSATION HISTORY:
{history_text if history_text else "(no previous messages)"}

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:"""

# -----------------------------
# AI GENERATION
# -----------------------------
def stream_ai(prompt: str):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
        stream=True,
    )
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def ask_ai(prompt: str) -> str:
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content or "Empty response."

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "DocMind API running", "status": "healthy"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/stats")
def stats():
    return {
        "active_sessions": len(session_history),
        "model": GROQ_MODEL,
        "vector_store": "Supabase pgvector",
    }

@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    session_history[session_id] = []
    return {"session_id": session_id}

@app.post("/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    """Upload and process a PDF. Chunks and embeddings stored in Supabase."""
    try:
        # Extract text
        pdf = PdfReader(file.file)
        full_text = "".join(page.extract_text() or "" for page in pdf.pages)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in PDF.")

        # Chunk text
        chunks = chunk_text(full_text)
        if not chunks:
            raise HTTPException(status_code=400, detail="Could not extract chunks from PDF.")

        # Embed chunks
        embeddings = await get_embedding_batch(chunks)

        # Store in Supabase
        sources = [file.filename] * len(chunks)
        await store_chunks(session_id, chunks, sources, embeddings)

        # Initialize session history if needed
        if session_id not in session_history:
            session_history[session_id] = []

        return {
            "message": "PDF uploaded and processed successfully.",
            "filename": file.filename,
            "chunks": len(chunks),
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest):
    """Ask a question about uploaded documents."""
    # Check session has documents
    has_docs = await session_has_documents(data.session_id)
    if not has_docs:
        raise HTTPException(status_code=400, detail="No documents uploaded for this session.")

    # Embed query
    query_embedding = await get_embedding(data.question)

    # Search Supabase
    results = await search_chunks(data.session_id, query_embedding)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    chunks = [r["content"] for r in results]
    sources = [r["source"] for r in results]

    # Build context
    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, chunks)
    )

    # Get history
    history = get_history(data.session_id)

    # Build and send prompt
    prompt = build_prompt(context, history, data.question)
    answer = ask_ai(prompt)

    # Save history
    save_history(data.session_id, data.question, answer)

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "source_chunks": chunks,
        "session_id": data.session_id,
    }

@app.post("/ask-stream")
async def ask_stream(data: QuestionRequest):
    """Stream response for a question about uploaded documents."""
    # Check session has documents
    has_docs = await session_has_documents(data.session_id)
    if not has_docs:
        raise HTTPException(status_code=400, detail="No documents uploaded for this session.")

    # Embed query
    query_embedding = await get_embedding(data.question)

    # Search Supabase
    results = await search_chunks(data.session_id, query_embedding)

    if not results:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    chunks = [r["content"] for r in results]
    sources = [r["source"] for r in results]

    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, chunks)
    )

    history = get_history(data.session_id)
    prompt = build_prompt(context, history, data.question)

    # Save to history after stream (approximate — stream answer not captured)
    save_history(data.session_id, data.question, "[streamed response]")

    return StreamingResponse(stream_ai(prompt), media_type="text/plain")

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Clean up session data from Supabase and memory."""
    await delete_session_chunks(session_id)
    session_history.pop(session_id, None)
    return {"message": "Session deleted."}
