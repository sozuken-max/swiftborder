# GPU backend speed test (swiftbackend-gpu)

A temporary Cloud Run service, `swiftbackend-gpu`, running the same `camdetect` code as
`swiftbackend` on an NVIDIA L4, to measure how much faster `model=local` gets. The live service and
the Firebase site are not touched. Delete the service when the test is done.

Only `model=local` / `model=local-conf` (the in-container ONNX model) can get faster; Roboflow models
run on Roboflow's servers either way.

## How the code picks the GPU

`_get_local_session` in `camdetect/main.py` uses CUDA whenever onnxruntime offers it. The normal
image installs CPU `onnxruntime`, which never offers it; `camdetect/gpu.Dockerfile` installs
`onnxruntime-gpu`. On the first `model=local` request the service logs
`local model providers: [...]`. If the list starts with `CPUExecutionProvider`, CUDA failed to start
and the "GPU" service is running on CPU.

## Steps (Cloud Shell, project swiftborder)

1. Open Cloud Shell on project `swiftborder` and get the branch:

   ```bash
   git clone -b claude/vibrant-ride-jpogum https://github.com/sozuken-max/swiftborder.git
   cd swiftborder
   gcloud config set project swiftborder
   ```

2. Check GPU quota: Console -> IAM & Admin -> Quotas, filter "Total Nvidia L4 GPU allocation without
   zonal redundancy", region europe-west1. It needs to be 1 or more; if it is 0, request 1 and wait.

3. Build and push the GPU image (about 5-10 minutes, the CUDA wheels are large):

   ```bash
   gcloud builds submit camdetect --config=camdetect/gpu.cloudbuild.yaml
   ```

4. Deploy, copying the env vars from the live service so Roboflow and settings match:

   ```bash
   gcloud run services describe swiftbackend --region=europe-west1 \
     --format=export > /tmp/live.yaml   # look under env: for the variable names and values

   gcloud run deploy swiftbackend-gpu \
     --image=europe-west1-docker.pkg.dev/swiftborder/cloud-run-source-deploy/swiftbackend-gpu:latest \
     --region=europe-west1 \
     --gpu=1 --gpu-type=nvidia-l4 --no-gpu-zonal-redundancy \
     --cpu=4 --memory=16Gi --no-cpu-throttling \
     --min-instances=0 --max-instances=1 --concurrency=4 \
     --no-allow-unauthenticated \
     --set-env-vars=ROBOFLOW_API_KEY=...   # plus any other env from /tmp/live.yaml
   ```

   `model=local` needs no Roboflow key, so for a local-only test the env vars can be skipped.

5. Check that it is really on the GPU:

   ```bash
   GPU_URL=$(gcloud run services describe swiftbackend-gpu --region=europe-west1 --format='value(status.url)')
   TOKEN=$(gcloud auth print-identity-token)
   curl -s -o /dev/null -w "%{http_code} %{time_total}s\n" -H "Authorization: Bearer $TOKEN" \
     "$GPU_URL/?format=directional&camera_id=2701&model=local"
   gcloud logging read 'resource.labels.service_name="swiftbackend-gpu" AND textPayload:"providers"' \
     --limit=3 --format='value(textPayload)'
   ```

   Expect `['CUDAExecutionProvider', 'CPUExecutionProvider']`.

6. Run the benchmark (same frames to both services):

   ```bash
   pip install requests
   python camdetect/bench_backends.py \
     --cpu https://swiftbackend-1095552466513.europe-west1.run.app/ \
     --gpu "$GPU_URL/" --token "$TOKEN" --n 30 \
     --out eval/runs/gpu-bench-$(date +%Y%m%d).csv
   ```

   The token expires after an hour; print a new one for longer runs.

7. Delete the service when finished (the L4 bills by the second while an instance is up):

   ```bash
   gcloud run services delete swiftbackend-gpu --region=europe-west1
   ```

## Reading the result

End-to-end latency includes fetching the LTA image and drawing the annotated JPEG, which the GPU does
not speed up. Smoke test against the CPU service on 2026-10-13: about 3 s warm, about 10 s cold.
Report cold start and warm p50/p95 separately, and whether the counts matched. No GPU number exists
until step 6 has run.
