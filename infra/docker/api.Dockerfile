# Cloud Run image for the owner-facing HTTP API only (`causalops.api_runtime:app`).
# The replay worker/lab stay VM-hosted -- see infra/gcp/cloud_run.tf's own
# scoping notes for why (lab/docker-compose.yml binds gateway/orders/inventory
# to 127.0.0.1 only, and shares a bind-mounted runs/ directory with the
# worker; neither survives a stateless container).
#
# Build from the REPO ROOT, not this directory, since it needs
# pyproject.toml/uv.lock/src/:
#   docker build -f infra/docker/api.Dockerfile -t causalops-api .

FROM python:3.12-slim

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
COPY src ./src
COPY README.md ./

# --no-dev: this image only ever runs the hosted API, never pytest/ruff/mypy.
RUN uv sync --locked --no-dev

ENV PYTHONUNBUFFERED=1
# Run the synced venv's own uvicorn directly -- `uv run` re-verifies the
# lockfile against the active dependency groups on every invocation, which
# silently pulled the dev group (mypy/ruff/...) back in at container start
# despite --no-dev above, adding tens of seconds before the app ever bound
# its port. Skips that resync entirely; the venv was already synced at
# build time and never changes at runtime.
ENV PATH="/app/.venv/bin:${PATH}"

# Cloud Run always sets PORT=8080 and routes traffic there.
EXPOSE 8080
CMD ["uvicorn", "causalops.api_runtime:app", "--factory", "--host", "0.0.0.0", "--port", "8080"]
