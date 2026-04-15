FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for geospatial packages (Rasterio, psycopg2, etc)
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    libpq-dev \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first (to keep image size strictly under 1GB for standard PaaS)
# IMPORTANT: Railway does not natively provision GPUs for standard deployments.
# CPU inference is utilized here. SNN inference for small patches takes ~2-5s on CPU.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Cloud platforms will inject the PORT environment variable
EXPOSE 8000

# Start server
CMD ["python", "start_server.py"]
