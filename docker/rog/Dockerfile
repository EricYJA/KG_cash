FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3-pip python3.10-venv git \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

RUN pip install \
        torch==2.4.1 --index-url https://download.pytorch.org/whl/cu118

# transformers is held at 4.x: 5.x requires bitsandbytes>=0.46.1, which has no
# kernels for Pascal (GTX 10xx, sm_61). bitsandbytes==0.42.0 is the last release
# that runs 8-bit there, so transformers must stay on a version that accepts it.
RUN pip install \
        transformers==4.46.3 \
        peft==0.13.2 \
        accelerate==1.0.1 \
        bitsandbytes==0.42.0 \
        sentencepiece==0.2.1 \
        tiktoken==0.12.0 \
        protobuf==6.33.1 \
        fastapi==0.115.0 \
        uvicorn[standard]==0.32.0 \
        pydantic==2.9.2

WORKDIR /app
COPY server.py /app/server.py

ENV MODEL_ID=rmanluo/RoG \
    QUANTIZATION=8bit \
    PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
