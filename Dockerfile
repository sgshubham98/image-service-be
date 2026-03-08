FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[dev]"

# Install PyTorch with CUDA 12.4 (override index for torch)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124

# Copy application
COPY app/ app/
COPY tests/ tests/

# Create output directory
RUN mkdir -p /app/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
