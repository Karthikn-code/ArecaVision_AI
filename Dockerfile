# Use official Python 3.10 slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV, FPDF, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir pytest

# Copy application source code
COPY . .

# Run project setup to initialize directories and SQLite database
RUN python setup_project.py

# Expose Streamlit default port
EXPOSE 8501

# Healthcheck to verify Streamlit app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command to launch the Streamlit application
CMD ["streamlit", "run", "run.py", "--server.port=8501", "--server.address=0.0.0.0"]
