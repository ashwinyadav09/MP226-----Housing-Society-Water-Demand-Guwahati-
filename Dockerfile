FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for Docker build caching
COPY requirements.txt pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code, parameters, and champion model artifact
COPY src/ src/
RUN pip install --no-cache-dir -e .

COPY params.yaml .
COPY models/ models/
COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
