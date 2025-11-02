import os
import base64
from typing import Any, Dict

import runpod
import numpy as np
import cv2
import requests

from pipeline_server import initialize_models, analyze_image_to_report

# Paths can be overridden via env vars on RunPod
PART_MODEL_PATH = os.environ.get("PART_MODEL_PATH", "models/part_detector.pt")
DAMAGE_MODEL_PATH = os.environ.get("DAMAGE_MODEL_PATH", "models/damage_seg.pt")
DEVICE = os.environ.get("DEVICE")  # e.g., "cuda" or "cpu"

# Load models once per worker (warm/hot runs reuse this process)
part_model, damage_model = initialize_models(
    PART_MODEL_PATH, DAMAGE_MODEL_PATH, DEVICE
)


def _load_image_from_input(job_input: Dict[str, Any]) -> np.ndarray:
    """
    Accepts either:
      - job_input["image"]     (base64-encoded string, no data: header)
      - job_input["image_url"] (http/https)
    Returns: BGR np.ndarray or raises ValueError.
    """
    if isinstance(job_input.get("image"), str):
        try:
            raw = base64.b64decode(job_input["image"])
        except Exception as e:
            raise ValueError(f"Invalid base64 image: {e}") from e
        data = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid base64 image data.")
        return img

    url = job_input.get("image_url")
    if isinstance(url, str) and url.startswith(("http://", "https://")):
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        data = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content at image_url.")
        return img

    raise ValueError("Provide 'image' (base64) or 'image_url'.")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler. Expects:
      job = {
        "id": "...",
        "input": {
          "image": "<base64>" OR "image_url": "https://...",
          "include_masks": false,
          "conf_parts": 0.5,
          "conf_damage": 0.5
        }
      }
    Returns:
      {"ok": True, "report": [...]}
    """
    job_input = job.get("input") or {}
    include_masks = bool(job_input.get("include_masks", False))
    conf_parts = float(job_input.get("conf_parts", 0.5))
    conf_damage = float(job_input.get("conf_damage", 0.5))

    img = _load_image_from_input(job_input)

    report = analyze_image_to_report(
        img,
        part_model,
        damage_model,
        conf_parts=conf_parts,
        conf_damage=conf_damage,
        include_masks=include_masks,
    )

    return {"ok": True, "report": report}


runpod.serverless.start({"handler": handler})