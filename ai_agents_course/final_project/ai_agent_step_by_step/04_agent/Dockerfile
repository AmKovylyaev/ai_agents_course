FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY langflow_components/ ./langflow_components/
COPY data/ ./data/
COPY run.py .

# Create directories
RUN mkdir -p /app/workspace /app/rag_storage

# Expose port
EXPOSE 7860

# Set environment
ENV LANGFLOW_COMPONENTS_PATH=/app/langflow_components

# Run
CMD ["python", "run.py"]
