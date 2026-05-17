FROM paddlepaddle/paddle:3.3.0-gpu-cuda11.8-cudnn8.9

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    gcc \
    git \
    curl \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libxkbcommon-x11-0 \
    libx11-6 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

# Keep scipy pinned because you were already pinning it;
# avoid layering random manual package installs later.
RUN python3 -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    requests \
    scipy==1.15.3

# MinerU docs recommend the fuller [all] install for most users.
RUN python3 -m pip install --no-cache-dir "mineru[all]"

COPY app /app/app

RUN mkdir -p /app/data/uploads /app/data/processing /app/data/results

EXPOSE 8001

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]