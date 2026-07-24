FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir poetry==2.4.1

WORKDIR /workspaces/alaska2-steganalysis
COPY pyproject.toml poetry.lock ./

RUN poetry install \
    --with dev,notebook,optimization,generation \
    --extras cuda \
    --no-root \
    --no-interaction \
    --no-ansi
