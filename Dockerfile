FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/data/hf
WORKDIR /app
# CPU-only torch keeps the image far smaller than the default CUDA wheels.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[embed,plots]"
COPY docker/config.yaml /app/config.docker.yaml
# Run as an unprivileged user (uid 1000 matches the usual first user on Linux hosts, so files written
# to the mounted ./data stay editable; on Docker Desktop for Windows/macOS ownership does not matter).
RUN useradd --uid 1000 --create-home app && mkdir -p /app/data && chown -R app /app/data
USER app
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request as u; u.urlopen(u.Request('http://127.0.0.1:8765/api/job', headers={'Host': 'localhost'}), timeout=4)"
# Config, database, digests, reports, models and the model cache all live in /app/data (a volume).
CMD ["sh", "-c", "mkdir -p data && { [ -f data/config.yaml ] || cp config.docker.yaml data/config.yaml; } && exec paperscreen --config data/config.yaml app --bind 0.0.0.0 --no-browser"]
