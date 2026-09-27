FROM python:3.12-slim

WORKDIR /app

# requirements.txt pins the CPU-only torch/torchvision build for Linux via
# --extra-index-url + a platform marker, so this is the only install step
# needed -- no separate CUDA-avoidance dance.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ ./api/
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8000

# No internet access needed at runtime: the committed model artifact holds
# its own backbone weights (see src/model.py), and the LLM report step
# falls back to a template if Ollama isn't reachable from the container.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
