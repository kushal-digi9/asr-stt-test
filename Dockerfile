FROM python:3.10-slim

WORKDIR /app

# Speed up pip and reduce disk use
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    libsndfile1 \
    ffmpeg \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CUDA wheels first (separate layer, avoids resolver backtracking)
# Adjust versions if you change Python/CUDA. These match CUDA 12.1 wheels.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu121 \
    torch==2.3.1+cu121 torchaudio==2.3.1+cu121

# Copy requirements and install remaining Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/outputs data/input_samples

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
