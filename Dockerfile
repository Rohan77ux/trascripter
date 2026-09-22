FROM python:3.12-slim

WORKDIR /app

# Install system dependencies if required for some packages (like pika/grpc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the app files
COPY . .

# Set env vars for OMP issue
ENV KMP_DUPLICATE_LIB_OK=TRUE

# Streamlit port
EXPOSE 8501
