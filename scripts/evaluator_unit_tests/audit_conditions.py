#!/usr/bin/env python3
"""Audit every registered evaluator-unit-test condition without scores."""

import argparse
import json
from pathlib import Path

ROOT = Path("/data/lanyi/projects/cfpr-v2")
OUT = Path("/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2")
REQUIRED = [
    "run_meta.json", "dwpose.npz", "face.json", "dino.json", "emotion.json",
    "hand_joint_visibility.npz", "lipsync/gen_lipsync.json",
    "temporal/gen_temporal_context.npz", "pose/pose.npz",
]


def absent(path, required):
    """Return required artifact paths that are absent below *path*."""
    return [name for name in required if not (path / name).exists()]


def condition_status(normal, required=REQUIRED):
    """Classify a condition, including complete post-hoc retry directories."""
    missing = absent(normal, required)
    if not missing:
        return {"status": "complete", "path": str(normal), "missing": []}
    for retry in sorted(normal.parent.glob("evaluators_posthoc_retry_*")):
        if not absent(retry, required):
            return {"status": "retried", "path": str(normal), "retry_path": str(retry), "missing": []}
    return {"status": "missing", "path": str(normal), "missing": missing}


def audit():
    """Audit the fixed 8-person × 2-window registered Cartesian product."""
    cfg = json.loads((ROOT / "scripts/evaluator_unit_tests/config.json").read_text())
    levels = {
        "T1": [f"k_{k:+d}" for k in cfg["t1_shifts_24fps"]],
        "T2": ["swap_next"], "T3": ["freeze", "skip", "jump"],
        "T4": ([f"ev_{x:g}" for x in cfg["t4_flicker_ev"]] + ["hard_cut"] + [f"instant_ev_{x:g}" for x in cfg["t4_single_frame_ev"]]),
        "T5": ["correct", "wrong"], "T6": ["right_hand_gray"],
    }
    rows = []
    for sample in cfg["samples"]:
        for window in cfg["windows"]:
            for test, test_levels in levels.items():
                for level in test_levels:
                    result = condition_status(OUT / test / sample / window / level / "evaluators")
                    result.update(sample=sample, window=window, test=test, level=level)
                    rows.append(result)
    return {
        "expected_conditions": len(rows),
        "complete": sum(row["status"] == "complete" for row in rows),
        "retried": sum(row["status"] == "retried" for row in rows),
        "missing": [row for row in rows if row["status"] == "missing"], "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="posthoc_condition_audit_20261109.json")
    args = parser.parse_args()
    destination = OUT / args.output
    if destination.exists():
        raise FileExistsError(destination)
    result = audit()
    destination.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: result[key] for key in ("expected_conditions", "complete", "retried", "missing")}, indent=2))


if __name__ == "__main__":
    main()
