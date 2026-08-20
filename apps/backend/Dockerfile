# syntax=docker/dockerfile:1.7

# Stage 1: build a wheel for the backend and install its deps into a venv
# using uv, which is much faster than pip for workspace layouts.
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder

WORKDIR /app

# Copy just enough to resolve the workspace lock.
COPY pyproject.toml uv.lock ./
COPY packages/cv-pipeline/pyproject.toml packages/cv-pipeline/pyproject.toml
COPY apps/backend/pyproject.toml apps/backend/pyproject.toml

# Sync dependencies into a project-local venv. --no-install-project
# means we install deps but not the packages themselves yet, so this
# layer is cached independent of source changes.
RUN uv sync --frozen --no-install-project --no-dev

# Now copy the actual source and install the packages into the venv.
COPY packages/cv-pipeline packages/cv-pipeline
COPY apps/backend apps/backend
RUN uv sync --frozen --no-dev

# Stage 2: runtime image. We copy the venv from the builder so the
# final image has no build tooling, which cuts size.
FROM python:3.11-slim-bookworm AS runtime

WORKDIR /app

# Copy the prepared venv and the source. Python uses the interpreter
# from the venv, which already knows about cv-pipeline and the backend
# through uv's workspace install.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/packages /app/packages
COPY --from=builder /app/apps/backend /app/apps/backend

ENV PATH="/app/.venv/bin:$PATH" \
	PYTHONUNBUFFERED=1 \
	PYTHONDONTWRITEBYTECODE=1 \
	CV_PIPELINE_CACHE_DIR=/models

# Expose a standard port; compose will map to host.
EXPOSE 8000

# Healthcheck: used by compose's depends_on: condition: service_healthy.
# 40s start period covers first-time weight download + first CUDA
# kernel compile. Do not shorten this.
HEALTHCHECK --interval=10s --timeout=5s --start-period=120s --retries=5 \
	CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1


COPY apps/backend/scripts/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["uvicorn", "api.main:app", \
     "--app-dir", "/app/apps/backend/src", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
