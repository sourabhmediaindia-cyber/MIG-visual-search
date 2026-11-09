FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MIG_OCR_MODEL_DIR=/root/.cache/torch/easyocr

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY index.py ocr.py ocr_index.py search.py web_app.py ./
COPY static ./static

RUN mkdir -p /app/index_data /photos /root/.cache/torch

EXPOSE 8000

CMD ["python", "web_app.py", "--host", "0.0.0.0", "--port", "8000", "--index", "/app/index_data", "--scan-folder", "/photos"]
