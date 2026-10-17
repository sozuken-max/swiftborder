#!/usr/bin/env bash
# Copy eval mix modules into forecast-api shipped_eval/ (image build + isolated import tests).
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$root/scripts/vendor_forecast_mix_eval.py"
