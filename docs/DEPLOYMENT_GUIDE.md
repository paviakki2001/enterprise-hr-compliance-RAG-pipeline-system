# Deployment Guide (Aligned to Your Week 3 Notebook)

## Backend API
Your notebook already defines:
- `GET /health`
- `POST /query`

For deployment, the cleanest approach is to use:
- `backend_support/week4_backend_app.py` as the deployable backend
- `frontend/streamlit_app.py` as the user interface

## Render
### Backend service
Start command:
```bash
uvicorn backend_support.week4_backend_app:app --host 0.0.0.0 --port $PORT
```

### UI service
Start command:
```bash
streamlit run frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0
```

Set environment variables:
- `BACKEND_BASE_URL`
- `BACKEND_QUERY_PATH=/query`
- `HEALTH_PATH=/health`
- `EMBEDDING_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `LLM_MODEL_NAME`

## Railway
Use the same commands as above.

## Hugging Face Spaces
Use Streamlit for the frontend.
Point `BACKEND_BASE_URL` to your deployed FastAPI backend.

## Docker
Backend:
```bash
uvicorn backend_support.week4_backend_app:app --host 0.0.0.0 --port 8000
```

Frontend:
```bash
streamlit run frontend/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
```

## Recommended production split
- Service 1: FastAPI backend
- Service 2: Streamlit frontend

This is more stable than trying to combine both into one public process.
