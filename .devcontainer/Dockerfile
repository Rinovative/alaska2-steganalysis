# 1) Basis-Image
FROM python:3.11-slim

# 2) System-Tools & Libs (für jpegio)
RUN apt-get update \
    && apt-get install -y \
    build-essential \
    pkg-config \
    libjpeg-dev \
    libturbojpeg0-dev \
    libgl1-mesa-glx \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3) Poetry installieren
RUN pip install poetry

# 4) Arbeitsverzeichnis
WORKDIR /app

# 5) Cacheschicht für Abhängigkeiten
COPY pyproject.toml poetry.lock ./

# 6) Dependencies installieren
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi