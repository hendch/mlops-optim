FROM python:3.12-slim

WORKDIR /mlflow

# Minimal OS deps (optional but commonly useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir mlflow

EXPOSE 5000

CMD ["mlflow", "ui",
     "--backend-store-uri", "sqlite:///mlflow.db",
     "--host", "0.0.0.0",
     "--port", "5000"]
