FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.128.2 uvicorn==0.40.0
COPY mock_server.py .
CMD ["uvicorn", "mock_server:app", "--host", "0.0.0.0", "--port", "8000"]

