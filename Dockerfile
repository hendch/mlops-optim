FROM python:3.13-slim
WORKDIR /app
# Don't write .pyc files, don't buffer logs, and don't keep pip cache
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models/ ./models/
COPY dvc.yaml dvc.lock ./

EXPOSE 8000
# Start the FastAPI API (model server)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
