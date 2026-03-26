#!/bin/bash
# Export SAM3 TensorRT engines using the est-slam conda env (TRT 10.3.0).
#
# Engines are written to /media/orin/external/trt_engines/sam3/ to avoid
# filling the internal drive.
#
# This must be run once before running the TRT benchmark.
# Re-run if the sam3 checkpoint changes or TRT version changes.
#
# Usage:
#   ./export_engines.sh              # encoder + decoder, FP16
#   ./export_engines.sh --encoder-only
#   ./export_engines.sh --precision fp32

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_PYTHON="/home/orin/miniforge3/envs/est-slam/bin/python3"
ENV_LIB="/home/orin/miniforge3/envs/est-slam/lib"

SAM3_PATH="/home/orin/dev/GSSG/thirdparty/sam3"
SAM3_DS_PATH="$SCRIPT_DIR/sam3_deepstream"
PATCHES_PATH="$SCRIPT_DIR/patches/sam3/model"

CHECKPOINT="/home/orin/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
OUTPUT_DIR="/media/orin/external/trt_engines/sam3"

echo "=================================================="
echo " SAM3 TensorRT Engine Export"
echo " TRT version : 10.3.0 (est-slam env)"
echo " Output dir  : $OUTPUT_DIR"
echo " Checkpoint  : $CHECKPOINT"
echo "=================================================="

# Apply trt_export.py patch into the est-slam sam3 if not already present
TRT_EXPORT_DST="$SAM3_PATH/sam3/model/trt_export.py"
TRT_EXPORT_SRC="$PATCHES_PATH/trt_export.py"

if [ ! -f "$TRT_EXPORT_DST" ]; then
    echo "Copying trt_export.py patch -> $TRT_EXPORT_DST"
    cp "$TRT_EXPORT_SRC" "$TRT_EXPORT_DST"
else
    # Check if patch is already the current version (compare sizes as quick check)
    if ! cmp -s "$TRT_EXPORT_SRC" "$TRT_EXPORT_DST"; then
        echo "Updating trt_export.py patch -> $TRT_EXPORT_DST"
        cp "$TRT_EXPORT_SRC" "$TRT_EXPORT_DST"
    else
        echo "trt_export.py already up-to-date"
    fi
fi

mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting export (this takes 5-15 minutes for FP16)..."
echo ""

PYTHONPATH="$SAM3_PATH:$SAM3_DS_PATH:${PYTHONPATH}" \
LD_LIBRARY_PATH="$ENV_LIB:${LD_LIBRARY_PATH}" \
    "$ENV_PYTHON" "$SAM3_DS_PATH/scripts/export_engines.py" \
        --checkpoint "$CHECKPOINT" \
        --output-dir "$OUTPUT_DIR" \
        --precision fp16 \
        "$@"

echo ""
echo "Engines written to: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"/*.engine 2>/dev/null || echo "(no .engine files yet - check for errors above)"
