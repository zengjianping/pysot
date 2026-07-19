import os
import sys
import argparse
import torch
import subprocess
import numpy as np
from typing import Tuple, List, Dict, Optional

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    print("Warning: TensorRT Python API not available.")

from export import MODELS


class TRTLogger(trt.ILogger):
    def __init__(self):
        trt.ILogger.__init__(self)

    def log(self, severity, msg):
        if severity == trt.Logger.ERROR:
            print(f"[TRT ERROR] {msg}")
        elif severity == trt.Logger.WARNING:
            print(f"[TRT WARNING] {msg}")
        elif severity == trt.Logger.INFO:
            print(f"[TRT INFO] {msg}")


def export_onnx(model, model_name: str, output_path: str, device: str = "cuda"):
    """Export model to ONNX format"""
    model = model.to(device).eval()

    # Prepare dummy inputs based on model type
    from pysot.odtrack.models import ODTrackModel

    if isinstance(model, ODTrackModel):
        # ODTrack uses 4 template images and 1 search image
        z = [torch.zeros(1, 3, 192, 192, device=device)] * 4
        x = torch.zeros(1, 3, 384, 384, device=device)
        input_names = ["template_0", "template_1", "template_2", "template_3", "search"]
        output_names = ["pred_logits", "pred_boxes"]
        dynamic_axes = {
            "template_0": {0: "batch"}, "template_1": {0: "batch"},
            "template_2": {0: "batch"}, "template_3": {0: "batch"},
            "search": {0: "batch"}
        }
        example_inputs = (z, x)
    else:
        # Standard siamese trackers
        z = torch.zeros(1, 3, 127, 127, device=device)
        x = torch.zeros(1, 3, 255, 255, device=device)
        input_names = ["template", "search"]
        output_names = ["cls", "loc"]
        dynamic_axes = {
            "template": {0: "batch"},
            "search": {0: "batch"}
        }
        dynamic_axes = None
        example_inputs = (z, x)

    print(f"Exporting ONNX model to {output_path}")
    with torch.no_grad():
        torch.onnx.export(
            model,
            example_inputs,
            output_path,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=False
        )
    print(f"ONNX model saved to {output_path}")
    return output_path


def build_engine_from_onnx_python(onnx_path: str, engine_path: str,
                                  fp16: bool = False, int8: bool = False,
                                  max_batch_size: int = 1,
                                  workspace_size: int = 2 << 30):
    """Build TensorRT engine from ONNX file using Python API (TensorRT 10.x compatible)"""

    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT Python API is not available")

    print(f"TensorRT version: {trt.__version__}")

    logger = TRTLogger()
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX
    print(f"Parsing ONNX file: {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # Build engine config
    config = builder.create_builder_config()

    # TensorRT 10.x: Use memory_pool_limit instead of max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)

    if fp16:
        print("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)

    if int8:
        print("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        # Note: INT8 requires calibration data, which is not implemented here

    # Build optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()

    for i in range(network.num_inputs):
        input_tensor = network.get_input(i)
        input_shape = input_tensor.shape
        print(f"Input {i}: {input_tensor.name}, shape: {input_shape}")

        # Set dynamic shapes (min, opt, max)
        min_shape = tuple(input_shape)
        opt_shape = tuple(input_shape)
        max_shape = list(input_shape)
        max_shape[0] = max_batch_size
        max_shape = tuple(max_shape)

        profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)

    config.add_optimization_profile(profile)

    print("Building TensorRT engine... This may take a while.")

    # TensorRT 10.x: build_serialized_network returns bytes directly
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Failed to build engine")
        return None

    # Save engine
    print(f"Saving TensorRT engine to {engine_path}")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved successfully")
    return engine_path


def build_engine_from_onnx_trtexec(onnx_path: str, engine_path: str,
                                   fp16: bool = False, int8: bool = False,
                                   max_batch_size: int = 1,
                                   workspace_size: int = 2 << 30):
    """Build TensorRT engine from ONNX file using trtexec command line tool (TensorRT 10.x)"""

    # Check if trtexec is available
    try:
        subprocess.run(["trtexec", "--help"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise RuntimeError("trtexec not found. Please ensure TensorRT is installed and trtexec is in PATH")

    # Build trtexec command
    # TensorRT 10.x uses --memPoolSize instead of --workspace
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_size}",
    ]

    if fp16:
        cmd.append("--fp16")
        print("Enabling FP16 mode")

    if int8:
        cmd.append("--int8")
        print("Enabling INT8 mode")

    # Add verbose output
    cmd.append("--verbose")

    print(f"Running trtexec command:")
    print(" ".join(cmd))
    print("\nBuilding TensorRT engine... This may take a while.\n")

    try:
        subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ TensorRT engine saved successfully to {engine_path}")
        return engine_path
    except subprocess.CalledProcessError as e:
        print(f"\n✗ trtexec failed with return code {e.returncode}")
        return None


def build_engine_from_onnx(onnx_path: str, engine_path: str,
                          fp16: bool = False, int8: bool = False,
                          max_batch_size: int = 1,
                          workspace_size: int = 2 << 30,
                          use_trtexec: bool = False):
    """Build TensorRT engine from ONNX file"""

    if use_trtexec:
        print("Using trtexec command line tool")
        return build_engine_from_onnx_trtexec(
            onnx_path, engine_path, fp16, int8, max_batch_size, workspace_size
        )
    else:
        print("Using TensorRT Python API")
        return build_engine_from_onnx_python(
            onnx_path, engine_path, fp16, int8, max_batch_size, workspace_size
        )


def export_tensorrt(model_name: str, weights_path: str, output_path: str,
                   device: str = "cuda", fp16: bool = False, int8: bool = False,
                   max_batch_size: int = 1, workspace_size: int = 2 << 30,
                   use_trtexec: bool = False):
    """Export model to TensorRT format"""

    # Create model
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(MODELS.keys())}")

    model_params = MODELS[model_name]["model_params"]
    model = MODELS[model_name]["model_type"](model_params)
    print(f"Created model: {model_name}")

    # Load weights
    if weights_path:
        state = torch.load(weights_path, map_location="cpu")
        if "net" in state:
            state = state["net"]
        elif "state_dict" in state:
            state = state["state_dict"]
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f"Loaded weights from {weights_path}")
    else:
        print("Warning: No weights provided, using random initialization")

    model = model.to(device).eval()

    # Step 1: Export to ONNX
    onnx_path = output_path.replace(".trt", ".onnx").replace(".engine", ".onnx")
    if not onnx_path.endswith(".onnx"):
        onnx_path = output_path + ".onnx"

    export_onnx(model, model_name, onnx_path, device)

    # Step 2: Build TensorRT engine from ONNX
    if not output_path.endswith((".trt", ".engine")):
        output_path = output_path + ".engine"

    engine_path = build_engine_from_onnx(
        onnx_path, output_path,
        fp16=fp16, int8=int8,
        max_batch_size=max_batch_size,
        workspace_size=workspace_size,
        use_trtexec=use_trtexec
    )

    if engine_path:
        print(f"\n{'='*60}")
        print(f"✓ TensorRT export completed successfully!")
        print(f"{'='*60}")
        print(f"  ONNX model:      {onnx_path}")
        print(f"  TensorRT engine: {engine_path}")
        print(f"  Method:          {'trtexec' if use_trtexec else 'Python API'}")
        print(f"  Precision:       {'FP16' if fp16 else 'INT8' if int8 else 'FP32'}")
        print(f"{'='*60}")
    else:
        print(f"\n✗ TensorRT export failed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export PySOT models to TensorRT')
    parser.add_argument('--config', type=str, required=True,
                       help='Model config name (e.g., siamrpn_r50_l234_dwxcorr)')
    parser.add_argument('--snapshot', type=str, required=True,
                       help='Path to model weights')
    parser.add_argument('--output', type=str, required=True,
                       help='Output path for TensorRT engine')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda or cpu)')
    parser.add_argument('--fp16', action='store_true',
                       help='Enable FP16 precision')
    parser.add_argument('--int8', action='store_true',
                       help='Enable INT8 precision (requires calibration)')
    parser.add_argument('--max-batch-size', type=int, default=1,
                       help='Maximum batch size for dynamic batching')
    parser.add_argument('--workspace-size', type=int, default=2048,
                       help='Workspace size in MB for TensorRT')
    parser.add_argument('--use-trtexec', action='store_true',
                       help='Use trtexec command line tool instead of Python API')

    args = parser.parse_args()

    workspace_bytes = args.workspace_size * (1 << 20)  # Convert MB to bytes

    export_tensorrt(
        model_name=args.config,
        weights_path=args.snapshot,
        output_path=args.output,
        device=args.device,
        fp16=args.fp16,
        int8=args.int8,
        max_batch_size=args.max_batch_size,
        workspace_size=workspace_bytes,
        use_trtexec=args.use_trtexec
    )
