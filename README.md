# Week 4 Complete Package (Regenerated from Your Week 3 Notebook)

This package is regenerated specifically from your uploaded notebook:

**`week_3_Semantic_RAG_Orchestration_with_fast_API.ipynb`**

So the Week 4 files are aligned with your Week 3 backend design, including:

- **FastAPI endpoint:** `POST /query`
- **Health endpoint:** `GET /health`
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Reranker model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM model:** `google/flan-t5-base`

## What this package contains

### Frontend
- Streamlit UI connected to your Week 3 FastAPI contract
- Support for:
  - question input
  - top_k_retrieval
  - top_k_context
  - optional metadata filters
  - answer display
  - citations display
  - latency display

### Backend upgrade files
- Production-style FastAPI app derived from your notebook logic
- Logging
- Monitoring
- CORS support
- response timing middleware
- optional cache helpers

### Deployment
- Dockerfile
- Render config
- Railway config
- Hugging Face Space config

### Documentation
- deployment guide
- UI guide
- architecture overview
- final evaluation template
- submission checklist

## Expected repository alignment

Your Week 3 notebook expects paths similar to:

```text
hr-compliance-rag/
├── data/
│   ├── metadata.csv
│   └── processed/
│       ├── chunks.csv
│       └── chunks.parquet
├── vectorstore/
│   ├── faiss.index
│   ├── embeddings.npy
│   ├── metadata_with_chunks.csv
│   └── manifest.json
```

## Main difference from the previous generic package

This regenerated package is **not generic**.  
It is tailored to the API contract and model choices already used in your uploaded Week 3 notebook.

## Quick start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit UI
```bash
streamlit run frontend/streamlit_app.py
```

### 3. Run backend app
```bash
uvicorn backend_support.week4_backend_app:app --host 0.0.0.0 --port 8000
```

### 4. Test endpoints
```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What are the onboarding compliance requirements?","top_k_retrieval":8,"top_k_context":4,"filters":null}'
```

## Included reference
The original uploaded notebook is copied into this package under:

`reference/week_3_Semantic_RAG_Orchestration_with_fast_API.ipynb`
