FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend ./frontend
COPY backend_support ./backend_support
COPY docs ./docs
COPY evaluation ./evaluation
COPY reference ./reference
COPY .env.example ./.env.example

EXPOSE 8501
EXPOSE 8000

CMD ["streamlit", "run", "frontend/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
