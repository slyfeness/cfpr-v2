#!/usr/bin/env python3
"""Frozen reporting helpers for the second identity-injection pilot round.

This script reads generated artefacts only.  It does not invoke generation or
change an evaluator; its detector calls serve solely to make the required
per-frame contact-sheet crops reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis

from scripts.chain_tests.identity_pilot_measurements import candidate_metrics
from scripts.chain_tests.analyze_identity_pilot import is_candidate

WINDOW = slice(96, 144)
OUTSIDE = np.r_[0:72, 168:240]
PAIR = {"CRB100_001": "CRB100_002", "CRB100_025": "CRB100_050"}


def mean(x):
    x = np.asarray(x, dtype=float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def metric_row(clean_own, clean_donor, own, donor, sharpness_ratio):
    return {
        "donor_rise": mean(donor[WINDOW]) - mean(clean_donor[WINDOW]),
        "own_drop": mean(clean_own[WINDOW]) - mean(own[WINDOW]),
        "sharpness_ratio": float(sharpness_ratio),
        "outside_own_delta": mean(own[OUTSIDE]) - mean(clean_own[OUTSIDE]),
        "valid_face_frames": int(np.isfinite(own).sum()),
    }


def detector():
    app = FaceAnalysis(name="buffalo_l", root="/data/lanyi/tmp/insightface_root", providers=["CPUExecutionProvider"], allowed_modules=["detection"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def face_crop(frame, app, size=192):
    faces = app.get(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    if not faces:
        return np.zeros((size, size, 3), np.uint8), None
    face = max(faces, key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]))
    x0, y0, x1, y1 = np.round(face.bbox).astype(int)
    d = int(.15 * max(x1-x0, y1-y0)); h, w = frame.shape[:2]
    x0, y0, x1, y1 = max(0, x0-d), max(0, y0-d), min(w, x1+d), min(h, y1+d)
    return cv2.resize(frame[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_CUBIC), [int(x0), int(y0), int(x1), int(y1)]


def render_contacts(first, second, step3, output):
    output.mkdir(parents=True, exist_ok=True); app = detector()
    result = {}
    for sample, donor in PAIR.items():
        sources = {
            "clean": first/sample/"clean_a"/"frames_uint8.npy",
            "P1": first/sample/"P1"/"frames_uint8.npy",
            "P1b": second/sample/"P1b"/"frames_uint8.npy",
            "D75": second/sample/"D75"/"frames_uint8.npy",
            "D90": second/sample/"D90"/"frames_uint8.npy",
            "H50": second/sample/"H50"/"frames_uint8.npy",
        }
        donor_frames = np.load(step3/donor/"clean_a"/"frames_uint8.npy")
        donor_tile, donor_box = face_crop(donor_frames[120], app)
        rows = []
        for label, filename in sources.items():
            frames = np.load(filename); tiles = []
            for frame_id in (108, 120, 132):
                tile, _ = face_crop(frames[frame_id], app); tiles.append(tile)
            tiles.append(donor_tile)
            row = np.concatenate(tiles, axis=1)
            cv2.putText(row, label, (5, 24), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 0, 0), 2)
            rows.append(row)
        sheet = np.concatenate(rows, axis=0)
        out = output/f"{sample}_round2_faces.png"
        cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        result[sample] = {"file": str(out), "donor_clean_frame": 120, "donor_clean_box": donor_box}
    return result


def render_dose_curve(data, output):
    """Render the fixed exploratory p=.5/.75/.9/1 dose curve.

    P4 at .5 is retained solely as a differently-transitioned reference point,
    exactly as prescribed by the task card.
    """
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), sharex=True)
    metrics = (("own_drop", "own ArcFace drop"), ("donor_rise", "donor ArcFace rise"), ("sharpness_ratio", "sharpness v2 / donor clean"))
    for row, sample in enumerate(PAIR):
        values = data["round1_v2_posthoc"][sample]["P4"]
        points = [values, data["round2"][sample]["D75"], data["round2"][sample]["D90"], data["round2"][sample]["P1b"]]
        for col, (key, title) in enumerate(metrics):
            ax = axes[row, col]
            ax.plot((.5, .75, .9, 1), [x[key] for x in points], marker="o")
            ax.axvline(.5, color="0.75", lw=1, ls="--")
            ax.set_title(f"{sample}: {title}")
            ax.set_xticks((.5, .75, .9, 1))
            ax.set_xlabel("replacement fraction p (P4 at .5 uses linear transition)")
            ax.grid(alpha=.25)
    fig.tight_layout()
    out = output / "dose_curves.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return str(out)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--step3", type=Path, required=True)
    p.add_argument("--v2-first", type=Path, required=True)
    p.add_argument("--v2-second", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    v1, v2 = json.loads(a.v2_first.read_text()), json.loads(a.v2_second.read_text())
    data = {"exploratory_unregistered": True, "window": [96, 144], "round1_v2_posthoc": {}, "round2": {}, "p1b_p6": {}}
    for sample in PAIR:
        clean = a.first/sample/"clean_a"/"evaluators"
        clean_own = np.load(clean/"arcface_own"/"buffalo_l_gen_sim.npy")
        clean_donor = np.load(clean/"arcface_donor"/"buffalo_l_gen_sim.npy")
        old = {}
        for variant in (f"P{i}" for i in range(8)):
            base = a.first/sample/variant/"evaluators"
            old[variant] = metric_row(clean_own, clean_donor, np.load(base/"arcface_own"/"buffalo_l_gen_sim.npy"), np.load(base/"arcface_donor"/"buffalo_l_gen_sim.npy"), v1["samples"][sample]["variants"][variant]["donor_clean_ratio"])
        data["round1_v2_posthoc"][sample] = old
        now = {}
        for variant in ("P1b", "D75", "D90", "H50"):
            base = a.second/sample/variant/"evaluators"
            now[variant] = metric_row(clean_own, clean_donor, np.load(base/"arcface_own"/"buffalo_l_gen_sim.npy"), np.load(base/"arcface_donor"/"buffalo_l_gen_sim.npy"), v2["samples"][sample]["variants"][variant]["donor_clean_ratio"])
        data["round2"][sample] = now
        p1b, p6 = now["P1b"], old["P6"]
        p1b_base = a.second/sample/"P1b"/"evaluators"
        p6_base = a.first/sample/"P6"/"evaluators"
        p1b_own = mean(np.load(p1b_base/"arcface_own"/"buffalo_l_gen_sim.npy")[WINDOW])
        p6_own = mean(np.load(p6_base/"arcface_own"/"buffalo_l_gen_sim.npy")[WINDOW])
        p1b_donor = mean(np.load(p1b_base/"arcface_donor"/"buffalo_l_gen_sim.npy")[WINDOW])
        p6_donor = mean(np.load(p6_base/"arcface_donor"/"buffalo_l_gen_sim.npy")[WINDOW])
        raw1 = json.loads((a.second/sample/"P1b"/"summary.json").read_text())["raw_output_sha256"]
        raw6 = json.loads((a.first/sample/"P6"/"summary.json").read_text())["raw_output_sha256"]
        comparison = {"p1b_window_own_arcface": p1b_own, "p6_window_own_arcface": p6_own, "own_arcface_delta_p1b_minus_p6": p1b_own-p6_own, "p1b_window_donor_arcface": p1b_donor, "p6_window_donor_arcface": p6_donor, "donor_arcface_delta_p1b_minus_p6": p1b_donor-p6_donor, "raw_output_sha256_equal": raw1 == raw6, "p1b_sha256": raw1, "p6_sha256": raw6}
        if raw1 != raw6:
            one = np.load(a.second/sample/"P1b"/"frames_uint8.npy").astype(np.float32)
            six = np.load(a.first/sample/"P6"/"frames_uint8.npy").astype(np.float32)
            comparison["max_per_frame_mean_absolute_pixel_difference"] = float(np.abs(one-six).mean(axis=(1,2,3)).max())
        data["p1b_p6"][sample] = comparison
    data["round1_v2_candidate"] = {v: is_candidate({s: data["round1_v2_posthoc"][s][v] for s in PAIR}) for v in (f"P{i}" for i in range(8))}
    data["round2_candidate"] = {v: is_candidate({s: data["round2"][s][v] for s in PAIR}) for v in ("P1b", "D75", "D90", "H50")}
    data["contact_sheets"] = render_contacts(a.first, a.second, a.step3, a.out/"contact_sheets")
    data["dose_curve"] = render_dose_curve(data, a.out)
    (a.out/"round2_analysis.json").write_text(json.dumps(data, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
