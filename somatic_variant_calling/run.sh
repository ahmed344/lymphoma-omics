#!/usr/bin/env bash
set -euo pipefail

# Helper script to run the Snakemake workflow for somatic variant calling.
# Usage: ./run.sh [snakemake options]
#
# Prerequisites:
#   - conda env create -f ../envs/somatic_variant_calling.yml
#   - conda activate somatic_variant_calling

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Load config to get thread count
THREADS=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yml'))['threads'])")

# Run snakemake with default options
# Pass through any additional arguments
snakemake \
    --cores "${THREADS}" \
    --use-conda \
    --conda-frontend mamba \
    "$@"
