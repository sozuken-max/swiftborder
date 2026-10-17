# Shipped copies of the mix

Cloud Build runs [`scripts/vendor_forecast_mix_eval.sh`](../../../scripts/vendor_forecast_mix_eval.sh)
to copy the mix shims plus `warehouse/` and the `harness/` modules they import into this
directory before the image is packed. The image context is `services/forecastapi`, so those
files are not in the image unless that script runs. Do not edit copies here; the source is
`eval/`.
