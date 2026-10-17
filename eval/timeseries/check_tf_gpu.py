#!/usr/bin/env python3
"""Print TensorFlow version and whether a GPU (incl. DirectML on Windows) is visible."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        import tensorflow as tf
    except ImportError:
        print("TensorFlow not installed.", file=sys.stderr)
        print("Windows GPU: pip install -r requirements-tf-gpu-windows.txt", file=sys.stderr)
        print("CPU / Linux: pip install tensorflow", file=sys.stderr)
        return 2

    print("TensorFlow:", tf.__version__)
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    devices = tf.config.list_physical_devices()
    print("Devices:", devices)
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected — LSTM will run on CPU.")
        return 1

    with tf.device("/GPU:0"):
        a = tf.random.normal((512, 512))
        b = tf.matmul(a, a)
    print("GPU matmul OK, shape:", b.shape)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
