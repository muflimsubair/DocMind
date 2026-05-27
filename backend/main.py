from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pypdf import PdfReader
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import groq
import os
import uuid
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# -----------------------------
# ENV SETUP
# -----------------------------
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is required.")

groq_client = groq.Groq(api_key=GROQ_API_KEY)
GROQ_MODEL = "llama-3.1-8b-instant"

# -----------------------------
# CONFIG
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
MAX_HISTORY = 4

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

class SynthesizeRequest(BaseModel):
    session_id: str
    mode: str = "compare"

class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    source_chunks: list[str]
    session_id: str

# -----------------------------
# GLOBALS
# -----------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
sessions: dict = {}

# -----------------------------
# QUERY ROUTER
# -----------------------------
def classify_query(question: str) -> str:
    q = question.lower()

    if "compare" in q or "difference" in q:
        return "compare"
    elif "summarize" in q or "summary" in q:
        return "summarize"
    elif "disagree" in q or "conflict" in q:
        return "disagree"
    else:
        return "qa"

# -----------------------------
# PROMPT BUILDER
# -----------------------------
def build_prompt(context, history, question):
    return f"""
You are DocMind, an advanced AI document analyst.

YOUR GOAL:
Provide the most accurate, clear, and helpful answer using ONLY the given context.

CORE RULES:
- Use ONLY the provided context (no external knowledge)
- If the answer is not present, say:
  "The documents do not contain this information."
- Be clear, concise, and natural
- Avoid repetition and unnecessary detail
- Combine related ideas instead of listing everything

QUALITY GUIDELINES:
- Focus on the MOST important and relevant information
- Avoid generic or textbook-style explanations
- Prefer meaningful insights over raw definitions
- Write in a smooth, human-like way (not robotic)

STYLE RULES:
- Do NOT overuse headings unless helpful
- Avoid phrases like:
  "The document discusses..." or "The main purpose is..."
- Keep answers clean, readable, and conversational
- Adapt length based on the question

ADAPT TO QUESTION TYPE:

• SUMMARY → concise paragraph covering key ideas  
• KEY POINTS → short bullets with only important info  
• OVERVIEW → 3–4 line smooth professional summary  
• CONCLUSION → combine ideas into insight (no repetition)  
• COMPARISON → clear similarities & differences  
• EXPLANATION → simple, clear explanation  

SPECIAL RULES:

For IMPORTANT INFORMATION:
- Group related ideas
- Remove repetition
- Focus on high-value insights only

For OVERVIEW:
- Focus on main purpose + key concepts (e.g., RAG)
- Avoid generic AI explanations

Conversation:
{history if history else "(no previous messages)"}

Context:
{context}

User Question:
{question}

Answer:
"""
# -----------------------------
# AI CALL
# -----------------------------
def ask_ai(prompt: str) -> str:
    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content or "Empty response."
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def stream_ai(prompt: str):
    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            yield content

def refine_answer(answer):
    prompt = f"""
Improve this answer:
- Make it clearer
- Remove repetition
- Make it more insightful

Answer:
{answer}
"""
    return ask_ai(prompt)

# -----------------------------
# UTILS
# -----------------------------
def chunk_text(text: str):
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = text[i:i + CHUNK_SIZE].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def embed_texts(texts):
    return np.array(embedding_model.encode(texts)).astype("float32")


def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Call POST /session first.")
    return sessions[session_id]


def retrieve_chunks(session, question, top_k=TOP_K):
    q_emb = embed_texts([question])
    distances, indices = session["index"].search(q_emb, top_k)

    chunks, sources = [], []
    for i in indices[0]:
        if i < len(session["texts"]):
            chunks.append(session["texts"][i])
            sources.append(session["sources"][i])

    return chunks, sources

def rerank_chunks(question, chunks, sources):

    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(chunks, sources, scores),
        key=lambda x: x[2],
        reverse=True
    )

    top_chunks = [c for c, _, _ in ranked[:TOP_K]]
    top_sources = [s for _, s, _ in ranked[:TOP_K]]

    return top_chunks, top_sources

def format_history(history):
    lines = []
    for turn in history[-MAX_HISTORY:]:
        lines.append(f"User: {turn['question']}")
        lines.append(f"Assistant: {turn['answer']}")
    return "\n".join(lines)

# -----------------------------
# ROUTES
# -----------------------------
@app.get("/")
def home():
    return {"message": "DocMind API running", "sessions": len(sessions)}

@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/stats")
def stats():
    return {
        "active_sessions": len(sessions),
        "model": GROQ_MODEL
    }


@app.post("/session")
def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "index": None,
        "texts": [],
        "sources": [],
        "docs": [],
        "history": [],
    }
    return {"session_id": session_id}


@app.post("/upload")
async def upload_pdf(session_id: str, file: UploadFile = File(...)):
    session = get_session(session_id)

    try:
        pdf = PdfReader(file.file)
        full_text = "".join(page.extract_text() or "" for page in pdf.pages)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="No readable text")

        chunks = chunk_text(full_text)
        embeddings = embed_texts(chunks)

        if session["index"] is None:
            session["index"] = faiss.IndexFlatL2(embeddings.shape[1])

        session["index"].add(embeddings)
        session["texts"].extend(chunks)
        session["sources"].extend([file.filename] * len(chunks))

        if file.filename not in session["docs"]:
            session["docs"].append(file.filename)

        return {"message": "Uploaded", "chunks": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(data: QuestionRequest):
    session = get_session(data.session_id)

    if session["index"] is None:
        raise HTTPException(status_code=400, detail="No documents uploaded")

    # Query routing
    query_type = classify_query(data.question)
    if query_type != "qa":
        return await synthesize(SynthesizeRequest(session_id=data.session_id, mode=query_type))

    chunks, sources = retrieve_chunks(session, data.question)

    chunks, sources = rerank_chunks(data.question, chunks, sources)

    context = "\n\n".join(
        f"[Source: {src}]\n{chunk}"
        for src, chunk in zip(sources, chunks)
    )

    history = format_history(session["history"])
    prompt = build_prompt(context, history, data.question)

    answer = ask_ai(prompt)
    answer = refine_answer(answer)

    session["history"].append({
        "question": data.question,
        "answer": answer
    })

    return {
        "answer": answer,
        "sources": list(dict.fromkeys(sources)),
        "source_chunks": chunks,
        "session_id": data.session_id,
    }


@app.post("/ask-stream")
async def ask_stream(data: QuestionRequest):

    session = get_session(data.session_id)

    chunks, sources = retrieve_chunks(session, data.question)

    context = "\n\n".join(
        f"[Source: {s}]\n{c}" for s, c in zip(sources, chunks)
    )

    history = format_history(session["history"])
    prompt = build_prompt(context, history, data.question)

    return StreamingResponse(stream_ai(prompt), media_type="text/plain")


@app.post("/synthesize", response_model=AnswerResponse)
async def synthesize(data: SynthesizeRequest):
    session = get_session(data.session_id)

    if len(session["docs"]) < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 docs")

    doc_chunks = {doc: [] for doc in session["docs"]}

    for text, source in zip(session["texts"], session["sources"]):
        if len(doc_chunks[source]) < 4:
            doc_chunks[source].append(text)

    full_context = "\n\n".join(
        f"=== {doc} ===\n" + "\n".join(chunks)
        for doc, chunks in doc_chunks.items()
    )

    instructions = {
        "compare": "Compare similarities and differences",
        "summarize": "Summarize each document and overall",
        "disagree": "Find contradictions between documents",
    }

    prompt = f"""
You are an expert AI analyst.

Task: {instructions.get(data.mode)}

Rules:
- Be structured
- Mention document names
- Be specific

Context:
{full_context}
"""

    answer = ask_ai(prompt)
    answer = refine_answer(answer)

    return {
        "answer": answer,
        "sources": session["docs"],
        "source_chunks": [],
        "session_id": data.session_id,
    }
