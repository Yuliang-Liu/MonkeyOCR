# Use CUDA environment and install basic dependencies
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    libgl1-mesa-glx libglib2.0-0 git curl \
    wget poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Force create symbolic links (overwrite if exists)
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Create non-root user
RUN useradd -m -s /bin/bash appuser

# Upgrade pip and setuptools (no mirror)
RUN pip install --upgrade pip setuptools

# Install PyTorch (official order: PyTorch first)
ARG CUDA_VERSION=126
RUN pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION}

# Copy project code
COPY --chown=appuser:appuser . /app/MonkeyOCR

# Set working directory to /app/MonkeyOCR
WORKDIR /app/MonkeyOCR

# Install project itself
RUN pip install -e .

# Install inference backend dependencies
RUN pip install lmdeploy==0.8.0

# Only install huggingface_hub (no modelscope)
RUN pip install huggingface_hub

# Install PaddlePaddle and PaddleX for PP-DocLayout_plus-L support
RUN pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu${CUDA_VERSION}/
RUN pip install "paddlex[base]"

# Create model directory
RUN mkdir -p /app/MonkeyOCR/model_weight && chown -R appuser:appuser /app/MonkeyOCR/model_weight

# Download models during build (explicit model name)
RUN python tools/download_model.py -n MonkeyOCR

# Copy model download scripts
COPY --chown=appuser:appuser docker/download_models.sh /app/MonkeyOCR/
COPY --chown=appuser:appuser docker/entrypoint.sh /app/MonkeyOCR/
RUN chmod +x /app/MonkeyOCR/download_models.sh /app/MonkeyOCR/entrypoint.sh

# Apply LMDeploy patcher
ARG LMDEPLOY_PATCHED=false
RUN if [ "$LMDEPLOY_PATCHED" = "true" ]; then \
      python /app/MonkeyOCR/tools/lmdeploy_patcher.py patch; \
      echo "Successfully apply lmdeploy patch" \
    else \
      echo "LMDEPLOY_PATCHED is false, skipping patch"; \
    fi

# Switch to non-root user
USER appuser

# Default command: start through entrypoint script
CMD ["/app/MonkeyOCR/entrypoint.sh"]


