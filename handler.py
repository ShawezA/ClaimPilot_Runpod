import os
import base64
from typing import Any, Dict

import runpod
import numpy as np
import cv2
import requests

from pipeline_server import initialize_models, analyze_image_to_report

PART_MODEL_PATH = os.environ.get("PART_MODEL_PATH", "/app/models/part_detector.pt")
DAMAGE_MODEL_PATH = os.environ.get("DAMAGE_MODEL_PATH", "/app/models/damage_seg.pt")
DEVICE = os.environ.get("DEVICE")  # "cuda" on GPU image

# Load once per worker
part_model, damage_model = initialize_models(
    PART_MODEL_PATH, DAMAGE_MODEL_PATH, DEVICE
)


def _load_image(job_input: Dict[str, Any]) -> np.ndarray:
    if "image" in job_input and isinstance(job_input["image"], str):
        raw = base64.b64decode(job_input["image"])
        data = np.frombuffer(raw, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid base64 image.")
        return img

    if "image_url" in job_input and isinstance(job_input["image_url"], str):
        r = requests.get(job_input["image_url"], timeout=20)
        r.raise_for_status()
        data = np.frombuffer(r.content, np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Invalid image content at image_url.")
        return img

    raise ValueError("Provide 'image' (base64) or 'image_url'.")


def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        job_input = job.get("input") or {}
        include_masks = bool(job_input.get("include_masks", True))
        conf_parts = float(job_input.get("conf_parts", 0.5))
        conf_damage = float(job_input.get("conf_damage", 0.5))

        img = _load_image(job_input)

        report = analyze_image_to_report(
            img,
            part_model,
            damage_model,
            conf_parts=conf_parts,
            conf_damage=conf_damage,
            include_masks=include_masks,
        )
        return {"ok": True, "report": report}
    except Exception as e:
        return {"ok": False, "error": str(e)}


runpod.serverless.start({"handler": handler})