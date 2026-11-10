#!/usr/bin/env python3
"""Analyze the pre-registered generator-side lip-offset matrix without retuning.

This script reads only completed evaluator JSON files.  It applies the thresholds
from docs/tasks/2026-11-08_step2_chain_lip_offset.md verbatim and writes a JSON
receipt suitable for the human-readable status report.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from scripts.chain_tests.lip_offset_config import build_registered_tasks


WINDOW_SEGMENTS = {"W1": range(8, 12), "W2": range(14, 18)}
WINDOW_FRAMES = {"W1": (96, 144), "W2": (168, 216)}
SHIFTS = (-8, -4, -2, 2, 4, 8)


def load_json(path: Path):
    return json.loads(path.read_text())


def task_dir(root: Path, task: dict) -> Path:
    return root / task["out_id"]


def source_lipsync(root: Path, task: dict) -> dict:
    path = task_dir(root, task) / "evaluators/lipsync/gen_lipsync.json"
    if not path.exists():
        raise FileNotFoundError(f"missing registered lip-sync output: {path}")
    return load_json(path)


def valid_delta(clean: dict, changed: dict, segment: int):
    base = clean["seg"][segment]
    edit = changed["seg"][segment]
    if base is None or edit is None or base["conf"] < 3:
        return None
    return float(edit["offset"] - base["offset"])


def median_or_none(rows):
    return statistics.median(rows) if rows else None


def analyze_window(clean: dict, changed: dict, name: str, k: int):
    inside = [valid_delta(clean, changed, segment) for segment in WINDOW_SEGMENTS[name]]
    inside = [value for value in inside if value is not None]
    start, end = WINDOW_FRAMES[name]
    outside = []
    for segment in range(20):
        seg_start, seg_end = segment * 12, (segment + 1) * 12
        if seg_end <= start - 6 or seg_start >= end + 6:
            value = valid_delta(clean, changed, segment)
            if value is not None:
                outside.append(value)
    expected = k * 25 / 24
    median = median_or_none(inside)
    return {
        "k_24fps": k,
        "expected_delta_25fps": expected,
        "inside_deltas_25fps": inside,
        "inside_n": len(inside),
        "inside_median_25fps": median,
        "G4_determinable": len(inside) >= 2,
        "G4_pass": None if len(inside) < 2 else abs(median - expected) <= 1.5,
        "outside_deltas_25fps": outside,
        "outside_n": len(outside),
        "G5_determinable": bool(outside),
        "G5_pass": None if not outside else sum(abs(value) <= 1 for value in outside) / len(outside) >= 0.9,
    }


def analyze(root: Path, config: dict) -> dict:
    tasks = build_registered_tasks(config["samples"], config["windows"], seed=config["seed"], frames=config["frames"])
    by_sample = {sample: [task for task in tasks if task["sample"] == sample] for sample in config["samples"]}
    people = {}
    for sample, sample_tasks in by_sample.items():
        named = {task["kind"]: task for task in sample_tasks}
        clean_a, clean_b, zero = named["clean_a"], named["clean_b"], named["zero_action"]
        clean_a_summary = load_json(task_dir(root, clean_a) / "summary.json")
        clean_b_summary = load_json(task_dir(root, clean_b) / "summary.json")
        zero_summary = load_json(task_dir(root, zero) / "summary.json")
        clean = source_lipsync(root, clean_a)
        windows = {}
        for window in ("W1", "W2"):
            windows[window] = {}
            for k in SHIFTS:
                changed = source_lipsync(root, named[f"{window}_k_{k:+d}"])
                windows[window][str(k)] = analyze_window(clean, changed, window, k)
        g6 = {}
        for window, rows in windows.items():
            g6[window] = {}
            for direction, ks in (("negative", (-2, -4, -8)), ("positive", (2, 4, 8))):
                medians = [rows[str(k)]["inside_median_25fps"] for k in ks]
                g6[window][direction] = {
                    "medians_abs_25fps": None if any(v is None for v in medians) else [abs(v) for v in medians],
                    "pass": None if any(v is None for v in medians) else abs(medians[0]) <= abs(medians[1]) <= abs(medians[2]),
                }
        people[sample] = {
            "low_confidence_in_unit_test": sample in {"CRB100_002", "CRB100_087"},
            "G1_clean_repeat": clean_a_summary["raw_output_sha256"] == clean_b_summary["raw_output_sha256"],
            "G2_zero_action": clean_a_summary["raw_output_sha256"] == zero_summary["raw_output_sha256"],
            "G3_exact_repair": clean_a_summary["raw_output_sha256"] == zero_summary["raw_output_sha256"],
            "clean_lipsync": {"seg_offsets": clean["seg_offsets"], "seg_conf": clean["seg_conf"]},
            "windows": windows,
            "G6": g6,
        }
    return {"analysis_spec": "pre_registered_2026-11-09", "people": people}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("scripts/chain_tests/config.json"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite analysis receipt: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analyze(args.root, load_json(args.config)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
