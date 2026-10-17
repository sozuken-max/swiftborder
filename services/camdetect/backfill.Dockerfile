# Cloud Run Job image for backfill_local_counts.py. Not used by swiftbackend (that is Dockerfile).
# Build context is camdetect/; see README "Retro-scoring stored frames".
FROM mirror.gcr.io/library/python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app
COPY requirements.txt requirements-backfill.txt ./
RUN pip install --no-cache-dir -r requirements-backfill.txt
COPY main.py backfill_local_counts.py ./
COPY models/ models/

ENTRYPOINT ["python", "backfill_local_counts.py"]
