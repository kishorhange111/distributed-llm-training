FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NCCL_DEBUG=INFO \
    TORCH_NCCL_ASYNC_ERROR_HANDLING=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts

RUN pip install --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -e . && \
    useradd --create-home --shell /bin/bash trainer && \
    chown -R trainer:trainer /app

USER trainer

ENTRYPOINT ["python", "-m", "disttrain.train"]
CMD ["--config", "configs/smoke.yaml"]
