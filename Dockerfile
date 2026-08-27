FROM python:3.11-slim

WORKDIR /app

RUN pip install \
    --no-cache-dir \
    --timeout 120 \
    --retries 10 \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements-serving.txt .

RUN pip install \
    --no-cache-dir \
    --timeout 120 \
    --retries 10 \
    -r requirements-serving.txt

COPY src/ ./src/
COPY release/ ./release/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]