FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    python3 \
    python3-pip \
    build-essential \
    ninja-build \
    python3-dev \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip

RUN git clone https://github.com/deepseek-ai/DeepSeek-OCR-2.git
WORKDIR /app/DeepSeek-OCR-2

RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu118

RUN pip install --no-cache-dir -r requirements.txt

ENV MAX_JOBS=4
RUN pip install --no-cache-dir flash-attn==2.7.3 --no-build-isolation

RUN pip install --no-cache-dir fastapi uvicorn python-multipart pymupdf pillow requests

ENV HF_HOME=/root/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

COPY api.py /app/DeepSeek-OCR-2/api.py

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
