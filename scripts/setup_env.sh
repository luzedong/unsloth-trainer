#!/usr/bin/env bash
# Setup training environment for Unsloth Trainer.
#
# Usage:
#   bash scripts/setup_env.sh              # Full install (default)
#   bash scripts/setup_env.sh --no-proxy   # Without GitHub proxy
#
# Prerequisites:
#   - Python 3.10+
#   - uv (pip install uv)
#   - CUDA 12.x + matching nvcc
set -euo pipefail

PROXY="https://hk.gh-proxy.org/https://github.com"
for arg in "$@"; do
    if [ "$arg" = "--no-proxy" ]; then
        PROXY="https://github.com"
    fi
done

echo "============================================"
echo "  Step 1/3: Core libraries + Unsloth"
echo "============================================"
uv pip install \
    "torch==2.8.0" \
    "triton>=3.3.0" \
    "numpy" \
    "pillow" \
    "torchvision" \
    "bitsandbytes" \
    "xformers==0.0.32.post2" \
    "unsloth_zoo[base] @ git+${PROXY}/unslothai/unsloth-zoo" \
    "unsloth[base] @ git+${PROXY}/unslothai/unsloth"

echo ""
echo "============================================"
echo "  Step 2/3: Pin trl / transformers versions"
echo "============================================"
uv pip install --upgrade --no-deps tokenizers trl==0.22.2 unsloth unsloth_zoo
uv pip install transformers==5.2.0

echo ""
echo "============================================"
echo "  Step 3/3: High-performance operators"
echo "============================================"
uv pip install --no-build-isolation flash-linear-attention causal_conv1d==1.6.0

# matplotlib for loss plots (optional, won't fail if skipped)
uv pip install matplotlib

echo ""
echo "============================================"
echo "  Done! Verify with:"
echo "    python -c \"import unsloth; print(unsloth.__version__)\""
echo "============================================"
