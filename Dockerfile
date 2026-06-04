FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ARG COOL_WHISPER_SNAPSHOT_DIR

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        ffmpeg \
        git \
        build-essential \
        python3-dev \
        ca-certificates \
        libc++1 \
        libc++abi1&& \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# timeout/retries for large torch wheels
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 --disable-pip-version-check install --timeout=120 --retries=5 \
        --index-url https://download.pytorch.org/whl/cu129 \
        torch torchvision torchaudio \
    || (echo "Initial install failed — retrying with extended timeout..." && \
        pip3 --disable-pip-version-check install --timeout=300 --retries=3 \
            --index-url https://download.pytorch.org/whl/cu129 \
            torch torchvision torchaudio)

COPY pyproject.toml README.md LICENSE boh.json ./
COPY whisperlivekit/ ./whisperlivekit/

# Editable install of whisperlivekit plus the optional extras
RUN pip3 install --no-cache-dir -e .[translation,translategemma_client,sentence_tokenizer]

# Additional dependencies
RUN pip3 install --no-cache-dir \
        safetensors \
        faster_whisper \
        huggingface_hub \
        yt-dlp \
        ten-vad \
        pillow \
	    wtpsplit


RUN mkdir -p /app/models /app/archives /app/ssl-config && \
    ln -sfn "$COOL_WHISPER_SNAPSHOT_DIR" /app/models/cool-whisper && \
    chmod 777 /app/archives

EXPOSE 8000

ENTRYPOINT ["python3", "/app/whisperlivekit/monitor-client/entrypoint.py"]
