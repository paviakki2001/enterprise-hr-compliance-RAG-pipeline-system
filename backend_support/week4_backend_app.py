import os
import re
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


# =========================================================
# APP SETUP
# =========================================================
app = FastAPI(
    title="HR Compliance RAG System",
    version="4.0.0",
    description="Week 4 Backend for HR Compliance RAG Assistant"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# GLOBALS
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DF: Optional[pd.DataFrame] = None
TEXT_COLUMN: Optional[str] = None


# =========================================================
# REQUEST / RESPONSE MODELS
# =========================================================
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question")
    top_k_retrieval: int = Field(default=5, ge=1, le=20)
    top_k_context: int = Field(default=3, ge=1, le=10)
    filters: Dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    status: str
    question: str
    answer: str
    retrieved_chunks: int
    sources: List[Dict[str, Any]]


# =========================================================
# HELPER FUNCTIONS
# =========================================================
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    cleaned = normalize_text(text)
    tokens = cleaned.split()
    stop_words = {
        "the", "is", "are", "a", "an", "of", "to", "for", "in", "on", "at", "by",
        "with", "from", "and", "or", "as", "be", "can", "this", "that", "it",
        "what", "which", "who", "when", "where", "how", "why", "does", "do",
        "did", "will", "shall", "should", "could", "would", "about", "into",
        "than", "then", "if", "but", "so", "any", "all"
    }
    return [t for t in tokens if t not in stop_words and len(t) > 1]


def find_chunks_file() -> Optional[Path]:
    possible_paths = [
        BASE_DIR / "metadata" / "chunks.csv",
        BASE_DIR / "metadata" / "metadata_chunks.csv",
        BASE_DIR / "metadata_chunks.csv",
        BASE_DIR / "data" / "chunks.csv",
        BASE_DIR / "data" / "metadata_chunks.csv",
        BASE_DIR / "artifacts" / "chunks.csv",
        BASE_DIR / "vectorstore" / "chunks.csv",
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Search recursively as fallback
    for path in BASE_DIR.rglob("*.csv"):
        if path.name.lower() in {"chunks.csv", "metadata_chunks.csv"}:
            return path

    return None


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "chunk_text",
        "text",
        "content",
        "page_content",
        "chunk",
        "document_text",
        "cleaned_text"
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_knowledge_base() -> None:
    global KNOWLEDGE_DF, TEXT_COLUMN

    csv_path = find_chunks_file()

    if csv_path is None:
        print("WARNING: No chunks CSV file found. UI will still run, but retrieval will be limited.")
        KNOWLEDGE_DF = None
        TEXT_COLUMN = None
        return

    try:
        df = pd.read_csv(csv_path)
        text_col = detect_text_column(df)

        if text_col is None:
            print(f"WARNING: CSV found at {csv_path}, but no valid text column detected.")
            KNOWLEDGE_DF = None
            TEXT_COLUMN = None
            return

        df[text_col] = df[text_col].fillna("").astype(str)
        df = df[df[text_col].str.strip() != ""].reset_index(drop=True)

        KNOWLEDGE_DF = df
        TEXT_COLUMN = text_col

        print(f"SUCCESS: Knowledge base loaded from: {csv_path}")
        print(f"Rows loaded: {len(KNOWLEDGE_DF)}")
        print(f"Detected text column: {TEXT_COLUMN}")

    except Exception as e:
        print(f"ERROR loading knowledge base: {e}")
        KNOWLEDGE_DF = None
        TEXT_COLUMN = None


def apply_filters(df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
    if df is None or df.empty or not filters:
        return df

    filtered_df = df.copy()

    for key, value in filters.items():
        if key in filtered_df.columns and value not in [None, "", []]:
            filtered_df = filtered_df[
                filtered_df[key].astype(str).str.lower() == str(value).lower()
            ]

    return filtered_df


def compute_score(question: str, chunk_text: str) -> float:
    q_norm = normalize_text(question)
    c_norm = normalize_text(chunk_text)

    q_tokens = set(tokenize(question))
    c_tokens = set(tokenize(chunk_text))

    if not c_norm:
        return 0.0

    overlap = len(q_tokens.intersection(c_tokens))
    token_score = overlap / (len(q_tokens) + 1)

    phrase_bonus = 0.0
    if q_norm and q_norm in c_norm:
        phrase_bonus += 3.0

    partial_bonus = 0.0
    for token in q_tokens:
        if token in c_norm:
            partial_bonus += 0.2

    length_penalty = 0.0
    if len(c_norm) > 2500:
        length_penalty = 0.2

    return token_score + phrase_bonus + partial_bonus - length_penalty


def retrieve_documents(question: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if KNOWLEDGE_DF is None or TEXT_COLUMN is None:
        return []

    df = KNOWLEDGE_DF.copy()
    df = apply_filters(df, filters or {})

    if df.empty:
        return []

    scored_docs = []
    for idx, row in df.iterrows():
        chunk_text = str(row.get(TEXT_COLUMN, ""))
        score = compute_score(question, chunk_text)

        if score > 0:
            row_dict = row.to_dict()
            row_dict["_score"] = float(score)
            row_dict["_row_index"] = int(idx)
            scored_docs.append(row_dict)

    scored_docs.sort(key=lambda x: x["_score"], reverse=True)
    return scored_docs[:top_k]


def split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_relevant_sentences(question: str, docs: List[Dict[str, Any]], max_sentences: int = 6) -> List[str]:
    q_tokens = set(tokenize(question))
    candidates = []

    for doc in docs:
        text = str(doc.get(TEXT_COLUMN, ""))
        for sent in split_into_sentences(text):
            sent_tokens = set(tokenize(sent))
            overlap = len(q_tokens.intersection(sent_tokens))
            if overlap > 0:
                candidates.append((overlap, sent))

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected = []
    seen = set()
    for _, sent in candidates:
        if sent not in seen:
            selected.append(sent)
            seen.add(sent)
        if len(selected) >= max_sentences:
            break

    return selected


def build_grounded_answer(question: str, docs: List[Dict[str, Any]], top_k_context: int = 3) -> str:
    if not docs:
        return (
            "I could not find enough relevant information in the available knowledge base to answer this question confidently.\n\n"
            "Possible reasons:\n"
            "• The topic may not be present in your chunks file.\n"
            "• The wording of the question may be too broad or unrelated to the stored documents.\n"
            "• The filters may be too restrictive.\n\n"
            "Try asking a more specific question related to the uploaded HR / compliance / policy content."
        )

    shortlisted_docs = docs[:top_k_context]
    relevant_sentences = extract_relevant_sentences(question, shortlisted_docs, max_sentences=6)

    if relevant_sentences:
        bullet_points = "\n".join([f"• {s}" for s in relevant_sentences[:6]])
        answer = (
            "Based on the retrieved knowledge base content, here is the most relevant answer:\n\n"
            f"{bullet_points}\n\n"
            "Note:\n"
            "• This response is grounded only in the retrieved chunks.\n"
            "• If you want a more precise answer, ask a narrower question."
        )
        return answer

    # fallback summary from chunks
    fallback_points = []
    for doc in shortlisted_docs:
        text = str(doc.get(TEXT_COLUMN, "")).strip()
        if text:
            short_text = text[:350].strip()
            fallback_points.append(f"• {short_text}")

    if fallback_points:
        return (
            "I found relevant documents, but could not extract highly focused sentences for the exact question.\n\n"
            "Most relevant retrieved content:\n\n"
            + "\n".join(fallback_points[:3])
        )

    return "Relevant documents were retrieved, but no usable answer could be constructed."


def build_sources(docs: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    sources = []

    for doc in docs[:limit]:
        source_item = {
            "score": round(float(doc.get("_score", 0.0)), 4),
            "file_name": str(doc.get("file_name", doc.get("source", "Unknown"))),
            "document_type": str(doc.get("document_type", "N/A")),
            "category": str(doc.get("category", "N/A")),
            "region": str(doc.get("region", "N/A")),
            "year": str(doc.get("year", "N/A")),
            "preview": str(doc.get(TEXT_COLUMN, ""))[:250].strip()
        }
        sources.append(source_item)

    return sources


# =========================================================
# STARTUP
# =========================================================
@app.on_event("startup")
def startup_event() -> None:
    load_knowledge_base()


# =========================================================
# ROUTES
# =========================================================
@app.get("/", response_class=JSONResponse)
def home():
    return {
        "message": "HR Compliance RAG System is running successfully.",
        "ui_url": "/ui",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health", response_class=JSONResponse)
def health_check():
    kb_status = "loaded" if KNOWLEDGE_DF is not None else "not_loaded"
    total_docs = int(len(KNOWLEDGE_DF)) if KNOWLEDGE_DF is not None else 0

    return {
        "status": "ok",
        "knowledge_base_status": kb_status,
        "total_chunks": total_docs,
        "text_column": TEXT_COLUMN
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(request: QueryRequest):
    try:
        question = request.question.strip()

        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        docs = retrieve_documents(
            question=question,
            top_k=request.top_k_retrieval,
            filters=request.filters
        )

        answer = build_grounded_answer(
            question=question,
            docs=docs,
            top_k_context=request.top_k_context
        )

        sources = build_sources(docs)

        return QueryResponse(
            status="success",
            question=question,
            answer=answer,
            retrieved_chunks=len(docs),
            sources=sources
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/ui", response_class=HTMLResponse)
def ui_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HR Compliance RAG Assistant</title>
    <style>
        * {
            box-sizing: border-box;
        }

        body {
            margin: 0;
            font-family: "Times New Roman", Times, serif;
            background:
                linear-gradient(135deg, rgba(8, 47, 73, 0.95), rgba(30, 64, 175, 0.90)),
                linear-gradient(45deg, rgba(124, 58, 237, 0.35), rgba(6, 182, 212, 0.25));
            min-height: 100vh;
            color: #f8fafc;
        }

        .page-wrapper {
            max-width: 1250px;
            margin: 0 auto;
            padding: 32px 20px 50px;
        }

        .hero {
            background: rgba(255, 255, 255, 0.12);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: 24px;
            padding: 28px;
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.25);
            backdrop-filter: blur(12px);
            margin-bottom: 24px;
        }

        .hero h1 {
            margin: 0 0 8px;
            font-size: 36px;
            color: #ffffff;
            letter-spacing: 0.5px;
        }

        .hero p {
            margin: 0;
            font-size: 18px;
            line-height: 1.7;
            color: #e2e8f0;
        }

        .grid {
            display: grid;
            grid-template-columns: 1.2fr 0.8fr;
            gap: 24px;
        }

        .card {
            background: rgba(255, 255, 255, 0.13);
            border: 1px solid rgba(255, 255, 255, 0.16);
            border-radius: 24px;
            padding: 24px;
            box-shadow: 0 18px 40px rgba(0, 0, 0, 0.22);
            backdrop-filter: blur(14px);
        }

        .section-title {
            margin-top: 0;
            margin-bottom: 18px;
            font-size: 26px;
            color: #fef3c7;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: bold;
            font-size: 18px;
            color: #ffffff;
        }

        textarea,
        input,
        select {
            width: 100%;
            border: 1px solid rgba(255, 255, 255, 0.28);
            border-radius: 16px;
            padding: 14px 16px;
            font-size: 17px;
            font-family: "Times New Roman", Times, serif;
            background: rgba(255, 255, 255, 0.92);
            color: #0f172a;
            outline: none;
            margin-bottom: 16px;
        }

        textarea {
            min-height: 180px;
            resize: vertical;
        }

        .row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
        }

        .button-row {
            display: flex;
            gap: 14px;
            flex-wrap: wrap;
            margin-top: 10px;
        }

        button {
            border: none;
            border-radius: 16px;
            padding: 14px 22px;
            font-size: 17px;
            font-family: "Times New Roman", Times, serif;
            font-weight: bold;
            cursor: pointer;
            transition: transform 0.18s ease, opacity 0.18s ease;
        }

        button:hover {
            transform: translateY(-2px);
            opacity: 0.95;
        }

        .primary-btn {
            background: linear-gradient(90deg, #f59e0b, #f97316);
            color: white;
            box-shadow: 0 10px 22px rgba(249, 115, 22, 0.30);
        }

        .secondary-btn {
            background: linear-gradient(90deg, #22c55e, #14b8a6);
            color: white;
            box-shadow: 0 10px 22px rgba(20, 184, 166, 0.30);
        }

        .tertiary-btn {
            background: linear-gradient(90deg, #8b5cf6, #ec4899);
            color: white;
            box-shadow: 0 10px 22px rgba(139, 92, 246, 0.30);
        }

        .answer-box,
        .status-box,
        .source-box {
            background: rgba(255, 255, 255, 0.94);
            color: #0f172a;
            border-radius: 18px;
            padding: 18px;
            min-height: 140px;
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 17px;
            border-left: 6px solid #f59e0b;
            overflow-wrap: break-word;
        }

        .status-box {
            min-height: auto;
            border-left-color: #22c55e;
            margin-bottom: 16px;
        }

        .source-box {
            border-left-color: #8b5cf6;
            min-height: auto;
            margin-bottom: 14px;
        }

        .badge {
            display: inline-block;
            padding: 7px 14px;
            border-radius: 999px;
            font-size: 15px;
            font-weight: bold;
            background: rgba(255, 255, 255, 0.16);
            color: #ffffff;
            margin-top: 10px;
        }

        .tips {
            margin: 0;
            padding-left: 22px;
            line-height: 1.8;
            font-size: 17px;
            color: #f8fafc;
        }

        .footer-note {
            margin-top: 18px;
            font-size: 15px;
            color: #dbeafe;
            line-height: 1.7;
        }

        @media (max-width: 900px) {
            .grid {
                grid-template-columns: 1fr;
            }

            .row {
                grid-template-columns: 1fr;
            }

            .hero h1 {
                font-size: 28px;
            }
        }
    </style>
</head>
<body>
    <div class="page-wrapper">
        <div class="hero">
            <h1>HR Compliance RAG Assistant</h1>
            <p>
                Intelligent question-answering interface for policy, compliance, SOP, labour law, and HR-related document retrieval.
                Ask meaningful questions in natural language. The system responds only from the available knowledge base context.
            </p>
            <div class="badge">Professional Week 4 Backend UI</div>
        </div>

        <div class="grid">
            <div class="card">
                <h2 class="section-title">Ask Your Question</h2>

                <label for="question">Enter your question</label>
                <textarea id="question" placeholder="Example: What are the employee leave compliance requirements for probationary staff?"></textarea>

                <div class="row">
                    <div>
                        <label for="top_k_retrieval">Top K Retrieval</label>
                        <input type="number" id="top_k_retrieval" value="5" min="1" max="20" />
                    </div>
                    <div>
                        <label for="top_k_context">Top K Context</label>
                        <input type="number" id="top_k_context" value="3" min="1" max="10" />
                    </div>
                </div>

                <div class="button-row">
                    <button class="primary-btn" onclick="askQuestion()">Submit Query</button>
                    <button class="secondary-btn" onclick="loadExample()">Load Example</button>
                    <button class="tertiary-btn" onclick="clearAll()">Clear</button>
                </div>

                <div class="footer-note">
                    This assistant can handle many different meaningful question styles, but answer quality depends on the available chunks in your dataset.
                    Best results come from clear, relevant, domain-related questions.
                </div>
            </div>

            <div class="card">
                <h2 class="section-title">Usage Guidance</h2>
                <ul class="tips">
                    <li>Ask specific HR, compliance, policy, SOP, legal, or internal process questions.</li>
                    <li>Use meaningful natural language questions.</li>
                    <li>If needed, ask narrower follow-up questions for better precision.</li>
                    <li>The response is grounded in retrieved document chunks only.</li>
                </ul>
            </div>
        </div>

        <div style="height: 24px;"></div>

        <div class="grid">
            <div class="card">
                <h2 class="section-title">System Response</h2>
                <div id="status" class="status-box">System ready. Enter a question and click Submit Query.</div>
                <div id="answer" class="answer-box">The answer will appear here.</div>
            </div>

            <div class="card">
                <h2 class="section-title">Retrieved Sources</h2>
                <div id="sources">
                    <div class="source-box">No sources retrieved yet.</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function loadExample() {
            document.getElementById("question").value =
                "What are the main compliance requirements for employee misconduct reporting and disciplinary procedure?";
        }

        function clearAll() {
            document.getElementById("question").value = "";
            document.getElementById("status").innerText = "System ready. Enter a question and click Submit Query.";
            document.getElementById("answer").innerText = "The answer will appear here.";
            document.getElementById("sources").innerHTML = '<div class="source-box">No sources retrieved yet.</div>';
        }

        async function askQuestion() {
            const question = document.getElementById("question").value.trim();
            const topKRetrieval = parseInt(document.getElementById("top_k_retrieval").value);
            const topKContext = parseInt(document.getElementById("top_k_context").value);

            if (!question) {
                document.getElementById("status").innerText = "Please enter a valid question.";
                return;
            }

            document.getElementById("status").innerText = "Processing your query...";
            document.getElementById("answer").innerText = "Please wait while the system retrieves relevant context and prepares the answer...";
            document.getElementById("sources").innerHTML = '<div class="source-box">Retrieving sources...</div>';

            try {
                const response = await fetch("/query", {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        question: question,
                        top_k_retrieval: topKRetrieval,
                        top_k_context: topKContext,
                        filters: {}
                    })
                });

                const data = await response.json();

                if (!response.ok) {
                    document.getElementById("status").innerText = "Query failed.";
                    document.getElementById("answer").innerText = data.detail || "Unknown error occurred.";
                    document.getElementById("sources").innerHTML = '<div class="source-box">No sources available.</div>';
                    return;
                }

                document.getElementById("status").innerText =
                    "Query completed successfully. Retrieved Chunks: " + data.retrieved_chunks;

                document.getElementById("answer").innerText = data.answer;

                if (data.sources && data.sources.length > 0) {
                    let sourcesHtml = "";
                    data.sources.forEach((src, index) => {
                        sourcesHtml += `
                            <div class="source-box">
                                <strong>Source ${index + 1}</strong><br>
                                <strong>File Name:</strong> ${src.file_name}<br>
                                <strong>Document Type:</strong> ${src.document_type}<br>
                                <strong>Category:</strong> ${src.category}<br>
                                <strong>Region:</strong> ${src.region}<br>
                                <strong>Year:</strong> ${src.year}<br>
                                <strong>Score:</strong> ${src.score}<br><br>
                                <strong>Preview:</strong><br>${src.preview}
                            </div>
                        `;
                    });
                    document.getElementById("sources").innerHTML = sourcesHtml;
                } else {
                    document.getElementById("sources").innerHTML =
                        '<div class="source-box">No relevant sources found.</div>';
                }

            } catch (error) {
                document.getElementById("status").innerText = "System error.";
                document.getElementById("answer").innerText =
                    "Unable to connect to the backend. Please ensure the FastAPI server is running.";
                document.getElementById("sources").innerHTML =
                    '<div class="source-box">No sources available due to connection failure.</div>';
            }
        }
    </script>
</body>
</html>
    """