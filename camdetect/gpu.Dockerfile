# swiftbackend-gpu image: same service as Dockerfile, with CUDA onnxruntime for a Cloud Run NVIDIA L4.
# The [cuda,cudnn] extra pulls the CUDA/cuDNN libraries as pip wheels, so a slim base is enough;
# Cloud Run mounts the NVIDIA driver into the container.
# Build from camdetect/: docker build -f gpu.Dockerfile -t IMAGE .
FROM mirror.gcr.io/library/python:3.11-slim@sha256:0dd364ba7e10242f07755449e3a3d0e35f9efd987952737b90def6709ab0c5ce

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt
COPY main.py .
COPY models/ models/

CMD exec functions-framework --target=detect --port=${PORT:-8080}
