FROM python:3.12-slim
LABEL maintainer="Luce Innovative Technologies"

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PORT="8080"
ENV HOST="0.0.0.0"
ENV LOG_LEVEL=INFO

WORKDIR /app
EXPOSE ${PORT}

COPY requirements.txt .

RUN apt update && apt install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt remove -y build-essential \
    && apt autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r luceit && adduser --system --no-create-home luceit \
    && chown -R luceit:luceit /app

COPY main.py ./
COPY app ./app
RUN chown -R luceit:luceit /app

USER luceit

CMD ["/bin/sh", "-c", "exec uvicorn --host ${HOST:-0.0.0.0} --port ${PORT:-8080} main:app"]