FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt ./

# Force CPU-only torch first, avoids pulling multi-GB CUDA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# Bake in the embedding model so the build doesn't need HF network access at runtime
COPY model_cache/local_llm/ ./model_cache/local_llm/
COPY . ./
CMD ["python", "app.py"]