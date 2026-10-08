# Cloud Run swiftbackend image, built by camdetect/cloudbuild.yaml. Same base image as the Test step
# (already on the build machine by the time this builds), so no buildpack builder download.
FROM mirror.gcr.io/library/python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py .

# Cloud Run sets PORT; functions-framework serves the `detect` HTTP function with gunicorn.
CMD exec functions-framework --target=detect --port=${PORT:-8080}
