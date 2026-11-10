#!/usr/bin/env python3
"""Apply the pre-registered identity-drift criteria without retuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.chain_tests.identity_config import ALPHAS, SAMPLES, WINDOWS, build_tasks


def _mean(values):
    values = np.asarray(values, dtype=float)
    return None if not len(values) else float(np.mean(values))


def analyze_window(clean_own, changed_own, clean_donor, changed_donor, window):
    """Compare one registered edit to clean at exactly the registered ranges."""
    start, end = window
    frame = np.arange(len(clean_own))
    inside = (frame >= start) & (frame < end)
    exterior = (frame < start - 24) | (frame >= end + 24)
    own_valid = np.isfinite(clean_own) & np.isfinite(changed_own)
    donor_valid = np.isfinite(clean_donor) & np.isfinite(changed_donor)
    own_delta = changed_own - clean_own
    donor_delta = changed_donor - clean_donor
    return {
        "window": [int(start), int(end)],
        "valid_inside_frames": int((inside & own_valid).sum()),
        "valid_inside_donor_frames": int((inside & donor_valid).sum()),
        "valid_exterior_frames": int((exterior & own_valid).sum()),
        "inside_own_clean": _mean(clean_own[inside & own_valid]),
        "inside_own_changed": _mean(changed_own[inside & own_valid]),
        "inside_own_delta": _mean(own_delta[inside & own_valid]),
        "inside_donor_clean": _mean(clean_donor[inside & donor_valid]),
        "inside_donor_changed": _mean(changed_donor[inside & donor_valid]),
        "inside_donor_delta": _mean(donor_delta[inside & donor_valid]),
        "outside_own_delta": _mean(own_delta[exterior & own_valid]),
        "transition_own_delta": _mean(own_delta[~inside & ~exterior & own_valid]),
    }


def dose_monotonic(drops, reversal_allowance=0.02):
    """Registered rule: any adjacent downward reversal may be at most 0.02."""
    return all(next_drop >= this_drop - reversal_allowance for this_drop, next_drop in zip(drops, drops[1:]))


def _array(root: Path, task: dict, reference: str):
    path = root / task["out_id"] / "evaluators" / f"arcface_{reference}" / "buffalo_l_gen_sim.npy"
    if not path.exists():
        raise FileNotFoundError(f"missing registered ArcFace output: {path}")
    return np.load(path)


def _summary(root: Path, task: dict):
    path = root / task["out_id"] / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"missing registered generation summary: {path}")
    return json.loads(path.read_text())


def _registered_tasks():
    tasks = build_tasks()
    return {sample: {task["kind"]: task for task in tasks if task["sample"] == sample} for sample in SAMPLES}


def _judgement(value, threshold, relation):
    if value is None:
        return None
    return value <= threshold if relation == "le" else value > threshold


def analyze(root: Path):
    people = {}
    registry = _registered_tasks()
    for sample, tasks in registry.items():
        clean = tasks["clean_a"]
        clean_own, clean_donor = _array(root, clean, "own"), _array(root, clean, "donor")
        hashes = {kind: _summary(root, task)["raw_output_sha256"] for kind, task in tasks.items() if kind in {"clean_a", "clean_b", "zero_action"}}
        windows = {}
        for window_name, window in WINDOWS.items():
            rows = {}
            for alpha in ALPHAS:
                task = tasks[f"{window_name}_a_{alpha:.2f}"]
                rows[f"{alpha:.2f}"] = analyze_window(clean_own, _array(root, task, "own"), clean_donor, _array(root, task, "donor"), window)
            windows[window_name] = rows

        g4_rows = [windows[window][f"{alpha:.2f}"] for window in WINDOWS for alpha in (.20, .40)]
        g4_determinable = all(row["valid_inside_frames"] >= 24 for row in g4_rows) and all(windows[window]["0.40"]["valid_inside_donor_frames"] >= 24 for window in WINDOWS)
        g4_pass = None if not g4_determinable else all(row["inside_own_delta"] <= -.10 for row in g4_rows) and all(windows[window]["0.40"]["inside_donor_delta"] > 0 for window in WINDOWS)
        g5_rows = [windows[window][f"{alpha:.2f}"] for window in WINDOWS for alpha in ALPHAS]
        g5_determinable = all(row["valid_exterior_frames"] > 0 for row in g5_rows)
        g5_pass = None if not g5_determinable else all(abs(row["outside_own_delta"]) <= .02 for row in g5_rows)
        g6_determinable = all(windows[window][f"{alpha:.2f}"]["valid_inside_frames"] >= 24 for window in WINDOWS for alpha in ALPHAS)
        dose = {window: [-windows[window][f"{alpha:.2f}"]["inside_own_delta"] for alpha in ALPHAS] for window in WINDOWS}
        g6_pass = None if not g6_determinable else all(dose_monotonic(dose[window]) for window in WINDOWS)
        people[sample] = {
            "G1_clean_repeat": hashes["clean_a"] == hashes["clean_b"],
            "G2_zero_action": hashes["clean_a"] == hashes["zero_action"],
            "G3_exact_repair": hashes["clean_a"] == hashes["zero_action"],
            "hashes": hashes,
            "windows": windows,
            "dose_own_drop": dose,
            "G4_injection_effect": g4_pass,
            "G5_outside_stable": g5_pass,
            "G6_dose_monotonic": g6_pass,
            "determinate": {"G4": g4_determinable, "G5": g5_determinable, "G6": g6_determinable},
        }
    overall = {}
    for key, field in (("G4", "G4_injection_effect"), ("G5", "G5_outside_stable"), ("G6", "G6_dose_monotonic")):
        rows = [people[sample][field] for sample in SAMPLES if sample != "CRB100_001" and people[sample][field] is not None]
        overall[key] = {"passed": sum(rows), "determinate_people": len(rows), "minimum_determinate": 5, "registered_pass_requirement": "at_least_6_of_other_7", "overall_pass": len(rows) >= 5 and sum(rows) >= 6}
    overall["G1"] = {"passed": sum(person["G1_clean_repeat"] for person in people.values()), "required": 8, "overall_pass": all(person["G1_clean_repeat"] for person in people.values())}
    overall["G2"] = {"passed": sum(person["G2_zero_action"] for person in people.values()), "required": 8, "overall_pass": all(person["G2_zero_action"] for person in people.values())}
    overall["G3"] = {"description": "same registered comparison as G2", "passed": sum(person["G3_exact_repair"] for person in people.values()), "required": 8, "overall_pass": all(person["G3_exact_repair"] for person in people.values())}
    return {"analysis_spec": "pre_registered_2026-11-09_step3", "primary_arcface_pack": "buffalo_l", "people": people, "overall": overall}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite analysis receipt: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(analyze(args.root), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
