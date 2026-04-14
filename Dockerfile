# Use a professional Python slim image
FROM python:3.11-slim

# Install system dependencies (specifically libgomp1 for LightGBM)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure models directory exists
RUN mkdir -p models

# Railway provides the PORT environment variable
ENV PORT=8001
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV PYTHONUNBUFFERED=1

# Start the service
CMD ["sh", "-c", "uvicorn src.inference_service:app --host 0.0.0.0 --port ${PORT}"]
