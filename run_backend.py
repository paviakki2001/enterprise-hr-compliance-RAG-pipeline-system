import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend_support.week4_backend_app:app", host="0.0.0.0", port=8000, reload=False)
