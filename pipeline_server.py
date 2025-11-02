import os
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def initialize_models(
    part_model_path: str,
    damage_model_path: str,
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(part_model_path):
        raise FileNotFoundError(
            f"Part detector model not found: {part_model_path}"
        )
    if not os.path.exists(damage_model_path):
        raise FileNotFoundError(
            f"Damage segmenter model not found: {damage_model_path}"
        )

    part_model = YOLO(part_model_path)
    damage_model = YOLO(damage_model_path)

    part_model.to(device)
    damage_model.to(device)

    print(f"Models loaded on device: {device}")
    return part_model, damage_model


def analyze_image_to_report(
    image_bgr: np.ndarray,
    part_model: YOLO,
    damage_model: YOLO,
    conf_parts: float = 0.5,
    conf_damage: float = 0.5,
    include_masks: bool = False,
) -> List[Dict[str, Any]]:
    """
    Returns a list of damaged parts with coordinates and per-damage entries:
    [
      {
        "Bauteil": "Front-bumper",
        "Bauteil_Koordinaten_x1y1x2y2": [x1, y1, x2, y2],
        "Schäden": [
          {"Schadensart": "Dent", "Konfidenz": 0.91, ...}
        ]
      },
      ...
    ]
    """
    report: List[Dict[str, Any]] = []
    if image_bgr is None or image_bgr.size == 0:
        return report

    h, w = image_bgr.shape[:2]

    part_results = part_model.predict(
        image_bgr, conf=conf_parts, imgsz=640, verbose=False
    )
    if not part_results:
        return report

    res = part_results[0]

    if res.boxes is None or len(res.boxes) == 0:
        return report

    # Iterate detected parts
    for i in range(len(res.boxes)):
        cls_id = int(res.boxes.cls[i].item())
        # names is a dict: {class_id: "name"}
        part_name = res.names.get(cls_id, str(cls_id))

        xyxy = (
            res.boxes.xyxy[i].detach().cpu().numpy().astype(int).tolist()
        )
        x1, y1, x2, y2 = xyxy
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = image_bgr[y1:y2, x1:x2]
        damages_for_part: List[Dict[str, Any]] = []

        if crop.size != 0:
            predict_params = {"conf": conf_damage, "verbose": False}
            if include_masks:
                predict_params["retina_masks"] = True

            damage_results = damage_model.predict(crop, **predict_params)
            if damage_results:
                dr = damage_results[0]
                if dr.boxes is not None and len(dr.boxes) > 0:
                    for j in range(len(dr.boxes)):
                        dmg_cls_id = int(dr.boxes.cls[j].item())
                        dmg_name = dr.names.get(dmg_cls_id, str(dmg_cls_id))
                        dmg_conf = float(dr.boxes.conf[j].item())
                        dmg_obj: Dict[str, Any] = {
                            "Schadensart": dmg_name,
                            "Konfidenz": dmg_conf,
                        }
                        # Optional: include segmentation polygon points
                        if include_masks and dr.masks is not None:
                            try:
                                poly = dr.masks.xy[j]
                                # relative to the crop
                                dmg_obj["Maske_Polygon_relativ"] = (
                                    poly.astype(int).tolist()
                                )
                            except Exception:
                                pass
                        damages_for_part.append(dmg_obj)

        if damages_for_part:
            report.append(
                {
                    "Bauteil": part_name,
                    "Bauteil_Koordinaten_x1y1x2y2": [x1, y1, x2, y2],
                    "Schäden": damages_for_part,
                }
            )

    return report