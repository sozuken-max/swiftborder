#!/usr/bin/env bash
# Run the camdetect, Causeway, and eval test suites (and repo doc checks).
#
#   python3.11 -m venv .venv
#   .venv/bin/pip install -r camdetect/requirements-dev.txt -r Causeway/requirements-dev.txt -r eval/requirements-dev.txt
#   .venv/bin/pip install -r eval/requirements-notebook.txt   # only for --slow
#
# Usage: scripts/run_tests.sh [--slow] [--coverage] [suite ...]
#   suites: camdetect Causeway eval repo (default: all)
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
py="${PYTHON:-}"
if [[ -z "$py" ]]; then
  if [[ -x "$root/.venv/bin/python" ]]; then py="$root/.venv/bin/python"; else py="python3"; fi
fi
# Suites run from their own directory, so make a relative interpreter path absolute.
if [[ "$py" == */* && "$py" != /* ]]; then py="$(pwd)/$py"; fi

slow=0
coverage=0
suites=()
for arg in "$@"; do
  case "$arg" in
    --slow) slow=1 ;;
    --coverage) coverage=1 ;;
    *) suites+=("$arg") ;;
  esac
done
if [[ ${#suites[@]} -eq 0 ]]; then suites=(camdetect Causeway eval repo); fi

failed=()
for name in "${suites[@]}"; do
  if [[ "$name" == "repo" ]]; then dir="$root/scripts"; else dir="$root/$name"; fi
  [[ -d "$dir/tests" ]] || { echo "skip $name (no tests/)"; continue; }
  args=(-m pytest -q -p no:cacheprovider)
  [[ $slow -eq 1 ]] && args+=(-m "slow or not slow")
  [[ $coverage -eq 1 && "$name" != "repo" ]] && args+=(--cov=. --cov-report=term-missing:skip-covered)
  echo "=== $name"
  if ! (cd "$dir" && "$py" "${args[@]}"); then failed+=("$name"); fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "FAILED: ${failed[*]}"
  exit 1
fi
echo "All suites passed."
