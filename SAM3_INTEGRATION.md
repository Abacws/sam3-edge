# SAM3-Edge Integration Guide

Reference document for agents and developers integrating SAM3-Edge on this Jetson AGX Orin.

---

## System snapshot

| Item | Value |
|------|-------|
| Device | NVIDIA Jetson AGX Orin 64GB |
| JetPack | R36.4.7 (JetPack 6.x) |
| Internal drive | 57 GB — **99% full, ~660 MB free. Do not install packages here.** |
| External drive | `/media/orin/external` — 55 GB, 53 GB free. Use this for engines, data, envs. |
| Python env | `est-slam` conda env — Python 3.10, Jetson PyTorch 2.5 (CUDA), TensorRT 10.3.0 |
| Repo | `/media/orin/external/dev/sam3-edge` |

---

## Paths that matter

```
/media/orin/external/dev/sam3-edge/       ← repo root
    benchmark_fps.py                      ← FPS benchmark script
    run_benchmark.sh                      ← benchmark wrapper (sets env correctly)
    export_engines.sh                     ← (re)build TRT engines
    sam3_deepstream/                      ← main Python package
        sam3_deepstream/
            api/server.py                 ← FastAPI server + SAM3Runtime class
            api/routes/video.py           ← video processing endpoints
            api/services/video_processor.py ← frame-by-frame segmentation logic
            inference/async_trt_runtime.py  ← TRT engine wrappers (sync + async + CUDA graphs)
            inference/trt_runtime.py        ← SAM3TRTRuntime (encoder + decoder)
            export/encoder_export.py        ← ViT → ONNX → TRT
            export/decoder_export.py        ← mask decoder → TRT
            config.py                       ← all config dataclasses, from_env()
            utils/mask_utils.py             ← RLE encode/decode, visualization
    patches/sam3/model/trt_export.py      ← applied to est-slam sam3 for TRT export

/media/orin/external/trt_engines/sam3/   ← compiled TRT engines (TRT 10.3.0)
    sam3_encoder.engine                   ← ViT encoder, FP16, input (1,3,504,504)
    sam3_decoder.engine                   ← mask decoder, FP16, 1.2 MB

/home/orin/.cache/huggingface/hub/models--facebook--sam3/
    snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt   ← checkpoint

/home/orin/dev/GSSG/thirdparty/sam3/     ← sam3 Python package (est-slam fork)
    sam3/model_builder.py                ← build_sam3_image_model()
    sam3/model/sam3_image_processor.py   ← Sam3Processor class
    sam3/model/trt_export.py             ← patched in by export_engines.sh
```

---

## Environment setup — required every session

The `est-slam` env has a missing `LD_LIBRARY_PATH` for `libcusparseLt`. Always prepend it:

```bash
export LD_LIBRARY_PATH=/home/orin/miniforge3/envs/est-slam/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/orin/dev/GSSG/thirdparty/sam3:/media/orin/external/dev/sam3-edge/sam3_deepstream:$PYTHONPATH
```

Use the est-slam Python directly:
```bash
/home/orin/miniforge3/envs/est-slam/bin/python3 your_script.py
```

Or activate it:
```bash
conda activate est-slam
# then still set LD_LIBRARY_PATH above
```

---

## How the model works

```
Input: RGB image (any size, numpy uint8 HWC)
        │
        ▼  resize to 504×504, normalize (ImageNet stats)
┌─────────────────────────────────────┐
│  ViT Encoder  (TRT engine)          │  ~100 ms  (PyTorch: ~1250 ms)
│  input:  (1, 3, 504, 504) float32   │
│  output: (1, 1024, 36, 36) float32  │  ← image features, cached in `state`
└─────────────────────────────────────┘
        │  run encoder ONCE per frame, decoder as many times as needed
        ▼
┌─────────────────────────────────────┐
│  Mask Decoder  (TRT engine)         │  ~8 ms
│  + VETextEncoder (text prompt)      │
│  OR geometric prompt (box/point)    │
└─────────────────────────────────────┘
        │
        ▼
  state["masks"]   → (N, 1, H, W) float32   binary mask per object
  state["boxes"]   → (N, 4) float32         [x1, y1, x2, y2] pixels
  state["scores"]  → (N,)   float32         confidence 0–1
```

**Important**: call `set_image()` once per frame (encoder, expensive), then call
`set_text_prompt()` or `add_geometric_prompt()` as many times as needed on the
same state (decoder, cheap). This is how you run multiple prompts on one frame
without paying the encoder cost each time.

---

## Measured performance (Jetson AGX Orin 64GB, TRT 10.3.0, FP16)

| Mode | Latency | FPS |
|------|---------|-----|
| PyTorch only (no TRT) | ~1254 ms | ~0.8 |
| TRT FP16 sync | ~104 ms | ~9.6 |
| TRT FP16 + async triple-buffer | ~103 ms | ~9.7 |
| TRT FP16 + CUDA graphs | ~101 ms | ~9.9 |

The encoder engine was exported at 504×504 (not 1008×1008) because ONNX tracing
at full resolution requires ~40 GB compute graph. The encoder input is therefore
`(1, 3, 504, 504)` — the benchmark and `preprocess_for_trt()` handle this automatically.

---

## Running the benchmark

```bash
cd /media/orin/external/dev/sam3-edge

./run_benchmark.sh                              # PyTorch baseline
./run_benchmark.sh --use-trt                    # TRT sync
./run_benchmark.sh --use-trt --async-pipeline   # TRT + triple-buffered CUDA streams
./run_benchmark.sh --use-trt --use-cuda-graphs  # TRT + CUDA graph capture
./run_benchmark.sh --use-trt --num-images 200 --mode point
./run_benchmark.sh --use-trt --prompt "chair" --num-images 500
```

Images are loaded from `/media/orin/external/data/replica/office2/results` (2000 RGB
`frame*.jpg` files). They are preloaded into RAM before timing starts — external
drive I/O does not affect benchmark numbers.

---

## Integration Option 1 — REST API

Start the server:
```bash
LD_LIBRARY_PATH=/home/orin/miniforge3/envs/est-slam/lib:$LD_LIBRARY_PATH \
PYTHONPATH=/home/orin/dev/GSSG/thirdparty/sam3:/media/orin/external/dev/sam3-edge/sam3_deepstream \
SAM3_CHECKPOINT="/home/orin/.cache/huggingface/hub/models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt" \
/home/orin/miniforge3/envs/est-slam/bin/python3 -m sam3_deepstream.api.server
```

Endpoints:
```
GET  /health                      → model status, GPU memory, uptime
POST /api/v1/segment              → image + text_prompt → masks JSON
POST /segment                     → image + points/boxes → PNG overlay or JSON
POST /api/v1/video/process        → upload video, returns job_id
GET  /api/v1/video/job/{id}/status
GET  /api/v1/video/job/{id}/result
```

Image segmentation example:
```python
import requests

with open("frame.jpg", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/api/v1/segment",
        files={"file": f},
        data={"text_prompt": "chair", "confidence_threshold": "0.5"},
    )

result = resp.json()
# result["detections"][i]["bbox"]     → [x1, y1, x2, y2]
# result["detections"][i]["score"]    → 0.87
# result["detections"][i]["mask_rle"] → RLE dict {counts, size}
# result["inference_time_ms"]         → wall time including encode+decode
```

---

## Integration Option 2 — Direct Python import

```python
import sys, os
os.environ["LD_LIBRARY_PATH"] = (
    "/home/orin/miniforge3/envs/est-slam/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
)
sys.path.insert(0, "/home/orin/dev/GSSG/thirdparty/sam3")
sys.path.insert(0, "/media/orin/external/dev/sam3-edge/sam3_deepstream")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from PIL import Image
import numpy as np

CHECKPOINT = (
    "/home/orin/.cache/huggingface/hub/models--facebook--sam3"
    "/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)

# Load once at startup (~30s)
model = build_sam3_image_model(
    checkpoint_path=CHECKPOINT, device="cuda",
    eval_mode=True, load_from_HF=False,
)
processor = Sam3Processor(model, resolution=1008, device="cuda", confidence_threshold=0.5)

# Per frame
def segment(frame_rgb: np.ndarray, prompt: str) -> dict:
    """frame_rgb: HxWx3 uint8 numpy array in RGB order."""
    state = processor.set_image(Image.fromarray(frame_rgb))   # encoder ~100ms
    state = processor.set_text_prompt(prompt, state)           # decoder ~8ms
    return state   # state["masks"], state["boxes"], state["scores"]

# Multiple prompts on the same frame — encoder runs only once
def segment_multi(frame_rgb: np.ndarray, prompts: list) -> list:
    state_base = processor.set_image(Image.fromarray(frame_rgb))
    return [processor.set_text_prompt(p, state_base) for p in prompts]

# Point/box prompt (normalized 0–1 coords)
def segment_point(frame_rgb: np.ndarray, cx: float, cy: float) -> dict:
    state = processor.set_image(Image.fromarray(frame_rgb))
    # box format: [cx, cy, width, height] normalized 0–1
    state = processor.add_geometric_prompt([cx, cy, 0.05, 0.05], True, state)
    return state
```

---

## Integration Option 3 — ROS 2 node (Humble)

```python
#!/usr/bin/env python3
# Save as sam3_node.py in your ROS package, run with est-slam Python (see below)
import sys
sys.path.insert(0, "/home/orin/dev/GSSG/thirdparty/sam3")
sys.path.insert(0, "/media/orin/external/dev/sam3-edge/sam3_deepstream")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PIL import Image as PILImage
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

CHECKPOINT = (
    "/home/orin/.cache/huggingface/hub/models--facebook--sam3"
    "/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
)

class SAM3Node(Node):
    def __init__(self):
        super().__init__("sam3")
        self.bridge = CvBridge()
        self.prompt = self.declare_parameter("prompt", "object").value

        model = build_sam3_image_model(
            checkpoint_path=CHECKPOINT, device="cuda",
            eval_mode=True, load_from_HF=False,
        )
        self.processor = Sam3Processor(model, resolution=1008, device="cuda")
        self.get_logger().info(f"SAM3 ready — prompt: '{self.prompt}'")

        self.sub = self.create_subscription(
            Image, "/camera/color/image_raw", self.cb, 1
        )

    def cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        state = self.processor.set_image(PILImage.fromarray(frame))
        state = self.processor.set_text_prompt(self.prompt, state)
        # use state["masks"], state["boxes"], state["scores"] here

def main():
    rclpy.init()
    rclpy.spin(SAM3Node())

if __name__ == "__main__":
    main()
```

Launch:
```bash
LD_LIBRARY_PATH=/home/orin/miniforge3/envs/est-slam/lib:$LD_LIBRARY_PATH \
/home/orin/miniforge3/envs/est-slam/bin/python3 sam3_node.py \
    --ros-args -p prompt:="chair"
```

---

## Rebuilding TRT engines

Required if: TRT version changes, checkpoint changes, or engines are missing.

```bash
cd /media/orin/external/dev/sam3-edge
./export_engines.sh              # ~3–5 min, writes to /media/orin/external/trt_engines/sam3/
```

The script:
1. Copies `patches/sam3/model/trt_export.py` into the est-slam sam3 fork
2. Runs `export_engines.py` with `build_sam3_image_model` (not the PE variant)
3. Exports ViT → ONNX → TRT FP16 encoder, and mask decoder → TRT FP16
4. Outputs ~855 MB encoder + ~1.2 MB decoder to external drive

**Do not use the `.engine` files in `sam3_deepstream/engines/`** — those were built
with a different TRT version and will fail to deserialize with TRT 10.3.0.

---

## Known issues and gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| `ImportError: libcusparseLt.so.0` | Not on LD_LIBRARY_PATH | Prepend `/home/orin/miniforge3/envs/est-slam/lib` |
| `cannot import name 'build_sam3_hiera_l'` | est-slam sam3 fork uses different builder name | Use `build_sam3_image_model` instead |
| `Serialization assertion plan->header.magicTag failed` | TRT engine version mismatch | Run `./export_engines.sh` to rebuild for TRT 10.3.0 |
| `Module onnx is not installed` | Missing in est-slam env | `pip install onnx` (already done) |
| `cannot import name 'build_sam3_pe_model'` | PE backbone not in est-slam fork | Only use standard model; PE requires a different fork |
| Internal drive full | 660 MB free on `/dev/mmcblk0p1` | Install all packages to external drive or est-slam env |
| Engines at 504×504 not 1008×1008 | Default in `trt_export.py` to avoid 40GB OOM during tracing | Expected; re-export with `export_image_size=1008` on line 807 of `trt_export.py` if needed (Orin 64GB can handle it) |
