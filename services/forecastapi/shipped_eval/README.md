# Shipped copies of the mix

Cloud Build copies `eval/layer_b_fcm_mlp.py` and `eval/layer_b_fcm_xgb_mix.py` here
before the image is packed. The image context is `services/forecastapi`, so those
files are not in the image unless this copy runs. Do not edit copies in this
directory; the source is `eval/`.
