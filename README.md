

# HR Compliance RAG Assistant

A FastAPI-based Retrieval-Augmented Generation (RAG) system designed to answer compliance-related queries using a structured knowledge base.

## 🚀 Features
- Interactive web UI (`/ui`)
- FastAPI backend (`/query`, `/health`)
- CSV-based knowledge base
- Retrieved source display with metadata
- Configurable retrieval parameters (Top-K)

## 🏗️ Architecture
- Backend: FastAPI
- Data Source: CSV (chunked documents)
- Retrieval: Keyword-based similarity
- Deployment: Render

## 📂 Project Structure
backend_support/
data/chunks.csv
requirements.txt
render.yaml
▶️ Run Locally
pip install -r requirements.txt
uvicorn backend_support.week4_backend_app:app --host 0.0.0.0 --port 8000
🌐 Endpoints
UI → /ui
Health → /health
Query → /query
☁️ Deployment

Deployed using Render with:

uvicorn backend_support.week4_backend_app:app --host 0.0.0.0 --port $PORT
📌 Note

This project demonstrates a complete RAG pipeline including data ingestion, retrieval, and response generation with source attribution.

👩‍💻 Author

Pavithra Veerapathiran


---

