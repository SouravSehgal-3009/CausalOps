FROM python:3.12-slim

# Pinned here and in pyproject.toml's dev group. The image needs it to run the
# services; the dev group needs it so their unit tests can import them on the host.
RUN pip install --no-cache-dir prometheus-client==0.21.1

WORKDIR /app
COPY . /app

ENV PYTHONUNBUFFERED=1
ENV CAUSALOPS_RUNS=/runs
