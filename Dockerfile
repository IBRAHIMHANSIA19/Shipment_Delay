# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code and model artifacts
COPY . .

# Expose ports for both the API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command to run the API service
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
