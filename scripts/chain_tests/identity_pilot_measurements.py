"""Frozen metrics for the P0--P7 pilot; no evaluator model is altered."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from scripts.chain_tests.identity_pilot import WINDOW, pilot_specs


def _mean(values):
    values = np.asarray(values, dtype=float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")


def candidate_metrics(clean_own, pilot_own, clean_donor, pilot_donor, clean_sharpness, pilot_sharpness):
    """Compute exactly the four locked candidate quantities (NaN propagates)."""
    start, end = WINDOW
    outside = np.r_[0:start - 24, end + 24:len(clean_own)]
    clean_sharpness = np.broadcast_to(np.asarray(clean_sharpness, dtype=float), np.asarray(clean_own).shape)
    pilot_sharpness = np.broadcast_to(np.asarray(pilot_sharpness, dtype=float), np.asarray(pilot_own).shape)
    return {
        "donor_rise": _mean(np.asarray(pilot_donor)[start:end]) - _mean(np.asarray(clean_donor)[start:end]),
        "own_drop": _mean(np.asarray(clean_own)[start:end]) - _mean(np.asarray(pilot_own)[start:end]),
        "sharpness_ratio": _mean(pilot_sharpness[start:end]) / _mean(clean_sharpness[start:end]),
        "outside_own_delta": _mean(np.asarray(pilot_own)[outside]) - _mean(np.asarray(clean_own)[outside]),
    }


def face_series(frames, detector):
    sharpness, boxes = [], []
    for frame in frames:
        faces = detector.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not faces:
            sharpness.append(float("nan")); boxes.append(None); continue
        face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
        x0, y0, x1, y1 = np.round(face.bbox).astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(frame.shape[1], x1), min(frame.shape[0], y1)
        crop = frame[y0:y1, x0:x1]
        sharpness.append(float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()) if crop.size else float("nan"))
        boxes.append([int(x0), int(y0), int(x1), int(y1)])
    return np.asarray(sharpness), boxes


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, required=True); args = parser.parse_args()
    app = FaceAnalysis(name="buffalo_l", root="/data/lanyi/tmp/insightface_root", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    output = {"window": list(WINDOW), "outside_frames": "[0,72) union [168,240)", "samples": {}}
    for sample in ("CRB100_001", "CRB100_025"):
        clean_dir = args.root / sample / "clean_a"
        clean_frames = np.load(clean_dir / "frames_uint8.npy")
        clean_sharp, clean_boxes = face_series(clean_frames, app)
        clean_own = np.load(clean_dir / "evaluators/arcface_own/buffalo_l_gen_sim.npy")
        clean_donor = np.load(clean_dir / "evaluators/arcface_donor/buffalo_l_gen_sim.npy")
        rows = {}
        for spec in pilot_specs():
            directory = args.root / sample / spec["id"]
            frames = np.load(directory / "frames_uint8.npy")
            sharp, boxes = face_series(frames, app)
            values = candidate_metrics(clean_own, np.load(directory / "evaluators/arcface_own/buffalo_l_gen_sim.npy"), clean_donor, np.load(directory / "evaluators/arcface_donor/buffalo_l_gen_sim.npy"), clean_sharp, sharp)
            values.update({"valid_face_frames": int(np.isfinite(sharp).sum()), "boxes": boxes, "sharpness": sharp.tolist()})
            rows[spec["id"]] = values
        output["samples"][sample] = {"clean": {"valid_face_frames": int(np.isfinite(clean_sharp).sum()), "boxes": clean_boxes, "sharpness": clean_sharp.tolist()}, "variants": rows}
    (args.root / "pilot_measurements.json").write_text(json.dumps(output, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
