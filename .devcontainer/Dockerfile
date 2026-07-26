FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /

RUN set -eu; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libjpeg-dev \
        util-linux; \
    if [ ! -d /workspace ] || [ -L /workspace ]; then \
        echo "Expected /workspace to be an ordinary directory" >&2; \
        exit 1; \
    fi; \
    if ! command -v mountpoint >/dev/null 2>&1; then \
        echo "Cannot verify whether /workspace is a mount point" >&2; \
        exit 1; \
    fi; \
    if mountpoint -q /workspace; then \
        echo "Refusing to remove mounted /workspace" >&2; \
        exit 1; \
    fi; \
    if find /workspace -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then \
        echo "Refusing to remove non-empty /workspace" >&2; \
        exit 1; \
    fi; \
    rmdir /workspace; \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir poetry==2.4.1

WORKDIR /workspaces/alaska2-steganalysis
COPY pyproject.toml poetry.lock ./

RUN poetry install \
    --with dev,notebook,optimization,generation \
    --extras cuda \
    --no-root \
    --no-interaction \
    --no-ansi \
    --no-cache \
    && rm -rf /root/.cache/pip /root/.cache/pypoetry
