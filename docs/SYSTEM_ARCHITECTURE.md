# System Architecture Overview

## Based on your uploaded Week 3 notebook

### Models used
- Embedding model: `sentence-transformers/all-mpnet-base-v2`
- Reranker model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- LLM model: `google/flan-t5-base`

## Flow
User -> Streamlit UI -> FastAPI `/query` -> semantic retrieval -> metadata filtering -> context selection -> grounded prompt -> LLM -> answer + citations -> UI

## Components

### 1. Data layer
- processed chunks
- metadata columns
- vector metadata CSV
- optional saved embeddings
- FAISS index

### 2. Retrieval layer
- semantic retrieval from embeddings
- metadata-aware filtering
- top-k candidate selection

### 3. Generation layer
- grounded prompt
- flan-t5-base answer generation
- citation packaging

### 4. Monitoring layer
- request count
- request errors
- total latency
- retrieval latency
- generation latency
- rotating file logs

### 5. Deployment layer
- FastAPI backend
- Streamlit frontend
- cloud deployment configs
