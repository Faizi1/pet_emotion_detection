"""
Export fine-tuned PyTorch emotion model to ONNX for faster CPU inference.

ONNX (Open Neural Network Exchange) is a portable model format.
ONNX Runtime runs it with graph optimizations — typically 30-40% faster
than raw PyTorch on CPU and uses less memory.

Usage:
  python scripts/export_onnx.py
  python scripts/export_onnx.py --weights services/models/pet_emotion_b3.pth
  python scripts/export_onnx.py --weights services/models/pet_emotion_b3.pth --output services/models/pet_emotion_b3.onnx
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from services.emotion_model import build_efficientnet_b3, load_model_classes  # noqa: E402


def export_onnx(weights_path: str, output_path: str, opset: int = 17) -> str:
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    classes = load_model_classes(weights_path)
    model = build_efficientnet_b3(num_classes=len(classes))
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    dummy = torch.randn(1, 3, 224, 224)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=max(opset, 18),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={
            "input": {0: "batch"},
            "logits": {0: "batch"},
        },
        dynamo=False,
    )

    # Quick sanity check with ONNX Runtime
    import onnxruntime as ort

    session = ort.InferenceSession(
        output_path,
        providers=["CPUExecutionProvider"],
    )
    logits = session.run(
        None,
        {"input": dummy.numpy().astype(np.float32)},
    )[0]
    if logits.shape != (1, len(classes)):
        raise RuntimeError(f"Unexpected ONNX output shape: {logits.shape}")

    print(f"ONNX model exported to {output_path}")
    print(f"Classes: {classes}")
    print(f"Output shape: {logits.shape} (batch x {len(classes)} emotions)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export pet emotion model to ONNX")
    parser.add_argument(
        "--weights",
        default=os.path.join("services", "models", "pet_emotion_b3.pth"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join("services", "models", "pet_emotion_b3.onnx"),
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    export_onnx(args.weights, args.output, opset=args.opset)


if __name__ == "__main__":
    main()
