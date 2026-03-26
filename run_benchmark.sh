#!/bin/bash
# Run the SAM3-Edge FPS benchmark using the est-slam conda environment.
#
# The est-slam env has Python 3.10, Jetson PyTorch 2.5 (CUDA), and TensorRT.
# libcusparseLt is inside the env's lib/ so we prepend it to LD_LIBRARY_PATH.
#
# Usage:
#   ./run_benchmark.sh                              # PyTorch baseline (~1250 ms)
#   ./run_benchmark.sh --use-trt                    # TRT FP16 encoder (~35 ms target)
#   ./run_benchmark.sh --use-trt --async-pipeline   # TRT + triple-buffered streams
#   ./run_benchmark.sh --use-trt --use-cuda-graphs  # TRT + CUDA graph capture
#   ./run_benchmark.sh --num-images 200 --mode point
#   ./run_benchmark.sh --mode text --prompt "chair"
#
# Run ./export_engines.sh first to build TRT engines compatible with TRT 10.3.0.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PYTHON="/home/orin/miniforge3/envs/est-slam/bin/python3"
ENV_LIB="/home/orin/miniforge3/envs/est-slam/lib"

# SAM3 package from est-slam fork (has build_sam3_image_model)
SAM3_PATH="/home/orin/dev/GSSG/thirdparty/sam3"

# sam3_deepstream package from this repo
SAM3_DS_PATH="$SCRIPT_DIR/sam3_deepstream"

if [ ! -f "$ENV_PYTHON" ]; then
    echo "ERROR: est-slam env not found at $ENV_PYTHON"
    echo "       Check that the conda environment exists: conda env list"
    exit 1
fi

echo "=================================================="
echo " SAM3-Edge FPS Benchmark"
echo " Python  : $ENV_PYTHON"
echo " sam3    : $SAM3_PATH"
echo " Args    : $*"
echo "=================================================="

PYTHONPATH="$SAM3_PATH:$SAM3_DS_PATH:${PYTHONPATH}" \
LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH}" \
    "$ENV_PYTHON" "$SCRIPT_DIR/benchmark_fps.py" "$@"
