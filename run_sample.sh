#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Step 1: Expert discovery
python -m analysis_specialize.main --config configs/sample_config.yaml

# Step 2: Domain steering evaluation
python -m analysis_specialize.domain_steering --config configs/domain_steering_config.yaml

# Step 3: Expert extraction — build dense model for math domain
python -m analysis_specialize.extract_and_finetune --config configs/extract_config.yaml

