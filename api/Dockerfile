FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# C3: roda como usuário não-root
RUN addgroup --system appgroup && adduser --system --ingroup appgroup appuser

COPY main.py .
RUN chown -R appuser:appgroup /app

USER appuser

# C3: healthcheck da própria API
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# M6: Gunicorn com 2 workers + timeout para inferência do LLM
CMD ["gunicorn", "main:app", "-k", "uvicorn.workers.UvicornWorker", \
     "--workers", "2", "--bind", "0.0.0.0:8000", "--timeout", "120", \
     "--access-logfile", "-", "--error-logfile", "-"]
