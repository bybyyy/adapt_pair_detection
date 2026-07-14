#!/usr/bin/env bash

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
campaign_root="${1:-/ssd_data/boran.y/apt_pair_detection_exploratory_v2}"

env -u LD_LIBRARY_PATH PYTHONUNBUFFERED=1 \
    "$repo_root/.venv-apt/bin/python" \
    "$repo_root/train_apt_pair_models.py" \
    "$campaign_root/datasets/manifest.json" \
    --outdir "$campaign_root/results" \
    --models cnn \
    --epochs 10 \
    --patience 3 \
    --device cpu

(
    cd "$campaign_root/results"
    sha256sum \
        apt_cnn_main.pt \
        apt_pair_detection_benchmark.json \
        apt_pair_detection_report.md \
        > SHA256SUMS
)
