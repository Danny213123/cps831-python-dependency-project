FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /workspace/
COPY src /workspace/src

RUN python -m pip install --upgrade pip \
    && python -m pip install .

CMD ["sleep", "infinity"]
