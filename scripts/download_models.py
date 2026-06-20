"""
Download YOLOv8 model into services/models/ for deployment.

Usage:
  python scripts/download_models.py
"""

import os
import shutil


def download_yolo():
    from ultralytics import YOLO

    model_size = os.getenv("YOLO_MODEL_SIZE", "s")
    model_name = f"yolov8{model_size}.pt"
    output_dir = os.path.join("services", "models")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name)

    if os.path.exists(output_path):
        print(f"Model already exists at {output_path}")
        return output_path

    print(f"Downloading YOLOv8-{model_size}...")
    YOLO(model_name)

    candidates = [
        os.path.expanduser(f"~/.config/Ultralytics/{model_name}"),
        model_name,
    ]
    for src in candidates:
        if os.path.exists(src):
            shutil.copy(src, output_path)
            print(f"Model saved to {output_path}")
            return output_path

    print("Model downloaded but local path not found. Check ~/.config/Ultralytics/")
    return None


if __name__ == "__main__":
    download_yolo()
