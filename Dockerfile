FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code.
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY model/disease_info.json ./model/disease_info.json

# Trained artifacts are mounted or copied at runtime:
#   -v $(pwd)/model/artifacts:/app/model/artifacts
RUN mkdir -p model/artifacts

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
