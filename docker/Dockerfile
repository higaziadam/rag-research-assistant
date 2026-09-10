FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements-runtime.txt ./requirements-runtime.txt
RUN pip install --no-cache-dir -r requirements-runtime.txt

COPY src ./src
COPY data ./data
COPY evaluation ./evaluation

RUN addgroup --system --gid 10001 rag \
    && adduser --system --uid 10001 --ingroup rag --home /app rag \
    && mkdir -p /app/artifacts \
    && chown -R rag:rag /app

ENV PYTHONPATH=/app/src
EXPOSE 8000

USER rag

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8000/ready')"

CMD ["uvicorn", "multimodal_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
