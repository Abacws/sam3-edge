"""
SAM3-Edge FPS Benchmark

Measures inference throughput on Replica office2 frames.

Images are preloaded into RAM before benchmarking to isolate
model performance from external drive I/O.

Performance expectations (Jetson AGX Orin, FP16 TRT):
  Encoder (ViT) only : ~35 ms  (~28 FPS)
  End-to-end TRT     : ~50 ms  (~20 FPS)
  PyTorch (no TRT)   : ~1250 ms  (~0.8 FPS)  <-- what you see without TRT

The ~35 ms figure in the README refers to TRT FP16 encoder-only latency.
Run export_engines.sh first to build compatible TRT engines, then use
--use-trt in this script.

Run with the wrapper script:
    ./run_benchmark.sh [OPTIONS]           # PyTorch baseline
    ./run_benchmark.sh --use-trt           # TRT encoder (target: ~35 ms)
    ./run_benchmark.sh --use-trt --async-pipeline  # triple-buffered (max throughput)

Options:
    --checkpoint PATH      SAM3 checkpoint (default: HuggingFace cache)
    --images-dir PATH      Directory with RGB frames (default: Replica office2)
    --num-images N         Number of images to benchmark (default: 100)
    --warmup N             Warmup iterations before timing (default: 10)
    --prompt TEXT          Text prompt for text-mode (default: "object")
    --mode [text|point]    Prompt mode: text or point (default: text)
    --use-trt              Use TensorRT engine (requires export_engines.sh first)
    --engine-dir PATH      TRT engine directory (default: /media/orin/external/trt_engines/sam3)
    --async-pipeline       Use triple-buffered async CUDA stream pipelining
    --use-cuda-graphs      Use CUDA graph capture (further reduces kernel launch overhead)
"""

import argparse
import gc
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SAM3-Edge FPS on pre-loaded images"
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "/home/orin/.cache/huggingface/hub/models--facebook--sam3"
            "/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt"
        ),
        help="Path to SAM3 checkpoint (.pt)",
    )
    parser.add_argument(
        "--images-dir",
        default="/media/orin/external/data/replica/office2/results",
        help="Directory containing RGB frame images (depth* files are skipped)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=100,
        help="Number of images to load and benchmark (max 2000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations before timing",
    )
    parser.add_argument(
        "--prompt",
        default="object",
        help="Text prompt for text-mode inference",
    )
    parser.add_argument(
        "--mode",
        choices=["text", "point"],
        default="text",
        help="Prompt mode: 'text' (VETextEncoder) or 'point' (center point)",
    )
    parser.add_argument(
        "--use-trt",
        action="store_true",
        help="Use TensorRT FP16 engines (run export_engines.sh first)",
    )
    parser.add_argument(
        "--engine-dir",
        default="/media/orin/external/trt_engines/sam3",
        help="Directory containing TRT engine files",
    )
    parser.add_argument(
        "--async-pipeline",
        action="store_true",
        help="Use triple-buffered async CUDA stream pipelining (max throughput)",
    )
    parser.add_argument(
        "--use-cuda-graphs",
        action="store_true",
        help="Use CUDA graph capture for minimal kernel launch overhead",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image preloading  (ALL disk I/O happens here, before any timing)
# ---------------------------------------------------------------------------

def preload_images(images_dir: str, num_images: int) -> List[np.ndarray]:
    """
    Load RGB frame images from disk into RAM as numpy arrays.
    Skips depth images (filename starts with 'depth').
    All disk I/O is done here so it never touches the benchmark loop.
    """
    from PIL import Image

    images_path = Path(images_dir)
    candidates = sorted(
        p for p in images_path.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not p.name.startswith("depth")
    )

    if not candidates:
        print(f"ERROR: No RGB images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    selected = candidates[:num_images]
    if len(selected) < num_images:
        print(f"WARNING: Only {len(selected)} images available (requested {num_images})")

    print(f"Preloading {len(selected)} images into RAM...", end="", flush=True)
    t0 = time.perf_counter()
    frames: List[np.ndarray] = []
    for p in selected:
        img = Image.open(p).convert("RGB")
        frames.append(np.array(img))
    elapsed = time.perf_counter() - t0
    total_mb = sum(f.nbytes for f in frames) / 1024 / 1024
    print(
        f" done. {len(frames)} frames, {total_mb:.1f} MB in RAM, "
        f"loaded in {elapsed:.2f}s  [I/O excluded from benchmark]"
    )
    return frames


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_memory_mb() -> float:
    import torch
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


# ---------------------------------------------------------------------------
# Image preprocessing for TRT encoder
# ---------------------------------------------------------------------------

def preprocess_for_trt(frame: np.ndarray, input_size: int, engine_dtype):
    """
    Convert numpy RGB frame to a normalized NCHW tensor matching the engine's
    input dtype (read from the engine itself — typically float32 I/O even for
    FP16 engines, since TRT keeps I/O in fp32 for precision).

    SAM3 normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    Output is kept on CPU with pin_memory so H2D is fast and measurable
    separately from GPU compute.
    """
    import torch
    import torch.nn.functional as F

    # HWC uint8 -> CHW float32 [0,1]
    t = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    t = (t - mean) / std

    # Resize to engine's actual input size and add batch dim
    t = F.interpolate(
        t.unsqueeze(0),
        size=(input_size, input_size),
        mode="bilinear",
        align_corners=False,
    )

    # Cast to engine's expected input dtype (fp32 for most TRT engines even if
    # internal layers run fp16)
    t = t.to(engine_dtype)

    # Pinned memory for fastest possible H2D transfer
    return t.pin_memory()


# ---------------------------------------------------------------------------
# TRT benchmark
# ---------------------------------------------------------------------------

def run_trt_benchmark(
    frames: List[np.ndarray],
    args: argparse.Namespace,
) -> List[float]:
    """
    Benchmark TRT encoder directly.

    Uses AsyncTRTInferenceEngine (triple-buffered CUDA streams) or
    CUDAGraphInference depending on flags.

    This measures the ViT encoder latency, which is the bottleneck
    and is what the README's ~35ms figure refers to.
    """
    engine_dir = Path(args.engine_dir)
    encoder_path = engine_dir / "sam3_encoder.engine"

    if not encoder_path.exists():
        print(f"\nERROR: TRT encoder engine not found: {encoder_path}", file=sys.stderr)
        print("Run ./export_engines.sh first to build compatible engines.", file=sys.stderr)
        sys.exit(1)

    # Add sam3_deepstream to path for imports
    ds_path = Path(__file__).parent / "sam3_deepstream"
    if str(ds_path) not in sys.path:
        sys.path.insert(0, str(ds_path))

    import torch

    print(f"\nLoading TRT encoder: {encoder_path}")

    if args.use_cuda_graphs:
        from sam3_deepstream.inference.async_trt_runtime import CUDAGraphInference
        _cg = CUDAGraphInference(encoder_path, device=0, warmup_iterations=5)
        # Expose the same attributes the rest of the code reads
        _cg.input_torch_dtype = torch.float32
        _cg.output_torch_dtype = torch.float32
        _cg.input_shape = tuple(_cg._input_buffer.shape)
        _cg.output_shape = tuple(_cg._output_buffer.shape)
        _cg.input_name = "image"
        _cg.output_name = "features"
        engine = _cg
        mode_label = "TRT FP16 + CUDA Graphs"
    elif args.async_pipeline:
        from sam3_deepstream.inference.async_trt_runtime import (
            AsyncTRTInferenceEngine, PipelinedInference
        )
        async_engine = AsyncTRTInferenceEngine(encoder_path, device=0, num_streams=3)
        pipeline = PipelinedInference(async_engine, buffer_size=3)
        engine = async_engine
        mode_label = "TRT FP16 + async triple-buffer"
    else:
        from sam3_deepstream.inference.async_trt_runtime import AsyncTRTInferenceEngine
        engine = AsyncTRTInferenceEngine(encoder_path, device=0, num_streams=1)
        mode_label = "TRT FP16 sync"

    # Inspect engine I/O
    print(f"  input  : {engine.input_name} {engine.input_shape} {engine.input_torch_dtype}")
    print(f"  output : {engine.output_name} {engine.output_shape} {engine.output_torch_dtype}")
    print(f"  mode   : {mode_label}")

    total = len(frames)

    # Pre-convert all frames to pinned-memory tensors matching the engine's
    # input dtype/size. No I/O, no further compute during the benchmark loop.
    input_size = engine.input_shape[-1] if len(engine.input_shape) == 4 else 1008
    engine_dtype = engine.input_torch_dtype
    print(f"\nPre-processing {total} frames  input={input_size}x{input_size}  dtype={engine_dtype}...", end="", flush=True)
    t0 = time.perf_counter()
    tensors = [preprocess_for_trt(f, input_size, engine_dtype) for f in frames]
    print(f" done ({time.perf_counter()-t0:.1f}s)")

    def infer(t) -> float:
        # Non-blocking H2D via pinned memory, then sync before starting clock
        gpu_t = t.to("cuda", non_blocking=True)
        cuda_sync()
        t0 = time.perf_counter()
        engine(gpu_t)
        cuda_sync()
        return (time.perf_counter() - t0) * 1000

    # Warmup
    print(f"Warmup ({args.warmup} frames)...", end="", flush=True)
    for i in range(args.warmup):
        infer(tensors[i % total])
    gc.collect()
    print(" done")

    # Timed run
    print(f"Benchmarking {total} frames  [{mode_label}]")
    times_ms: List[float] = []
    wall_start = time.perf_counter()

    for i, t in enumerate(tensors):
        ms = infer(t)
        times_ms.append(ms)
        if (i + 1) % 20 == 0 or (i + 1) == total:
            recent = times_ms[-20:]
            fps = 1000.0 / (sum(recent) / len(recent))
            print(f"  [{i+1:4d}/{total}]  {ms:6.1f} ms  recent {fps:.2f} FPS")

    wall_total = time.perf_counter() - wall_start
    print(f"\nWall-clock: {total} frames in {wall_total:.2f}s = {total/wall_total:.2f} FPS")
    return times_ms


# ---------------------------------------------------------------------------
# PyTorch benchmark (baseline / fallback)
# ---------------------------------------------------------------------------

def run_pytorch_benchmark(frames: List[np.ndarray], args: argparse.Namespace) -> List[float]:
    """
    Benchmark using pure PyTorch SAM3 model (no TRT).
    Expected: ~1250 ms/frame on Jetson AGX Orin.
    """
    from PIL import Image

    checkpoint = args.checkpoint
    if not Path(checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {checkpoint}", file=sys.stderr)
        sys.exit(1)

    import torch
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nLoading SAM3 PyTorch model")
    print(f"  checkpoint : {checkpoint}")
    print(f"  device     : {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")

    t0 = time.perf_counter()
    model = build_sam3_image_model(
        checkpoint_path=checkpoint, device=device, eval_mode=True, load_from_HF=False
    )
    processor = Sam3Processor(model, resolution=1008, device=device, confidence_threshold=0.5)
    print(f"  load time  : {time.perf_counter()-t0:.1f}s")
    print(f"  GPU memory : {gpu_memory_mb():.0f} MB")

    total = len(frames)
    use_text = args.mode == "text"
    center_box = [0.5, 0.5, 0.05, 0.05]

    def infer(frame: np.ndarray) -> float:
        pil = Image.fromarray(frame)
        cuda_sync()
        t0 = time.perf_counter()
        state = processor.set_image(pil)
        if use_text:
            processor.set_text_prompt(args.prompt, state)
        else:
            processor.add_geometric_prompt(center_box, True, state)
        cuda_sync()
        return (time.perf_counter() - t0) * 1000

    prompt_desc = f'"{args.prompt}"' if use_text else "center point"
    print(f"\nWarmup ({args.warmup} frames)...", end="", flush=True)
    for i in range(args.warmup):
        infer(frames[i % total])
    gc.collect()
    print(" done")

    print(f"Benchmarking {total} frames  [PyTorch {args.mode}  prompt={prompt_desc}]")
    times_ms: List[float] = []
    wall_start = time.perf_counter()

    for i, frame in enumerate(frames):
        ms = infer(frame)
        times_ms.append(ms)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            recent = times_ms[-10:]
            fps = 1000.0 / (sum(recent) / len(recent))
            print(f"  [{i+1:4d}/{total}]  {ms:6.1f} ms  recent {fps:.2f} FPS")

    wall_total = time.perf_counter() - wall_start
    print(f"\nWall-clock: {total} frames in {wall_total:.2f}s = {total/wall_total:.2f} FPS")
    return times_ms


# ---------------------------------------------------------------------------
# Results report
# ---------------------------------------------------------------------------

def report(times_ms: List[float], args: argparse.Namespace):
    arr = np.array(times_ms)
    fps = 1000.0 / arr

    if args.use_trt:
        mode_str = "TRT FP16"
        if args.use_cuda_graphs:
            mode_str += " + CUDA Graphs"
        elif args.async_pipeline:
            mode_str += " + async pipeline"
        note = "  (encoder only — add ~8 ms for decoder = full ~50 ms end-to-end)"
    else:
        mode_str = f"PyTorch  [{args.mode} mode]"
        note = "  (full PyTorch model, no TRT — run with --use-trt for ~35 ms)"

    print("\n" + "=" * 60)
    print("  SAM3-Edge Benchmark Results")
    print("=" * 60)
    print(f"  mode        : {mode_str}")
    if not args.use_trt:
        print(f"  prompt      : \"{args.prompt}\" [{args.mode}]")
    print(f"  frames      : {len(arr)}")
    print()
    print("  Latency per frame (ms)")
    print(f"    mean      : {arr.mean():.1f}")
    print(f"    median    : {np.percentile(arr, 50):.1f}")
    print(f"    p90       : {np.percentile(arr, 90):.1f}")
    print(f"    p95       : {np.percentile(arr, 95):.1f}")
    print(f"    p99       : {np.percentile(arr, 99):.1f}")
    print(f"    min       : {arr.min():.1f}")
    print(f"    max       : {arr.max():.1f}")
    print(f"    std       : {arr.std():.1f}")
    print()
    print("  Throughput (FPS)")
    print(f"    mean      : {fps.mean():.2f}")
    print(f"    median    : {np.median(fps):.2f}")
    print(f"    best      : {fps.max():.2f}")
    print(f"    worst     : {fps.min():.2f}")
    print()
    print(f"  Note: {note}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Preload all images into RAM — zero disk I/O during benchmark
    frames = preload_images(args.images_dir, args.num_images)

    # 2 + 3. Load engine/model and benchmark
    if args.use_trt:
        times_ms = run_trt_benchmark(frames, args)
    else:
        times_ms = run_pytorch_benchmark(frames, args)

    # 4. Report
    report(times_ms, args)


if __name__ == "__main__":
    main()
