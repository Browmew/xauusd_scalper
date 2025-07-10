# Use Python 3.9 slim image for smaller size
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies required for scientific computing and plotting
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements first to leverage Docker build cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p data/historical/ticks data/historical/l2_orderbook data/live \
    configs models/trained reports logs

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Set Python path to include src directory
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Set entrypoint to main CLI application
ENTRYPOINT ["python", "src/main.py"]

# Default command (can be overridden)
CMD ["--help"]