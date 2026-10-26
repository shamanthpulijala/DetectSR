FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (models, static assets, backend code)
COPY . .

# Expose standard port
EXPOSE 8000
ENV PORT=8000
ENV TF_ENABLE_ONEDNN_OPTS=0

# Start unified FastAPI server serving both API and Frontend
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
