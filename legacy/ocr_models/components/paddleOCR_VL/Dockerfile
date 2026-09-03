FROM paddlepaddle/paddle:3.3.0-gpu-cuda11.8-cudnn8.9

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

# PaddleOCR-VL recommended install path: PaddleOCR + doc-parser extra
RUN python3 -m pip install --no-cache-dir -U "paddleocr[doc-parser]"

# API deps
RUN python3 -m pip install --no-cache-dir \
    fastapi uvicorn python-multipart pymupdf pillow requests

COPY api.py /app/api.py

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
