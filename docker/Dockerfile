# Multi-stage Dockerfile for FastAPI

# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY ../requirements/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy Python dependencies from builder stage
COPY --from=builder /root/.local /home/appuser/.local

# Copy only the FastAPI application
COPY ../src/fastapi_app.py .

# Copy AnomaVision package (required for the app)
COPY ../src/AnomaVision/anodet ./AnomaVision/anodet

# Create directories for model files
RUN mkdir -p distributions

# Copy model files if they exist (optional - can be mounted as volumes)
COPY ../padim_model.onnx . 
# COPY ../distributions/padim_model.pt ./distributions/

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Add local bin to PATH
ENV PATH=/home/appuser/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]

