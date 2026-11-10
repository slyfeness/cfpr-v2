#!/usr/bin/env python3
"""Run the unchanged ArcFace evaluator against own and original donor references."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.chain_tests.identity_config import DONOR, build_tasks


INPUT_ROOT = Path("/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1")
EVALUATOR = Path("scripts/evaluators/eval_identity_arcface.py")
PYTHON = Path("/data/lanyi/.conda/envs/dh-eval-prm/bin/python")


def arcface_command(root: Path, task: dict, reference_sample: str, reference_kind: str):
    task_dir = root / task["out_id"]
    return [
        str(PYTHON), str(EVALUATOR), str(INPUT_ROOT / reference_sample / "reference.png"),
        str(task_dir / "evaluators" / f"arcface_{reference_kind}"),
        f"gen={task_dir / 'frames_uint8.npy'}",
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    tasks = [task for task in build_tasks() if not args.samples or task["sample"] in args.samples]
    for task in tasks:
        for kind, ref_sample in (("own", task["sample"]), ("donor", DONOR[task["sample"]])):
            output = args.root / task["out_id"] / "evaluators" / f"arcface_{kind}" / "buffalo_l_gen_sim.npy"
            command = arcface_command(args.root, task, ref_sample, kind)
            if output.exists():
                print({"task": task["out_id"], "reference": kind, "status": "already_present"}, flush=True)
            elif args.dry_run:
                print({"task": task["out_id"], "reference": kind, "command": command}, flush=True)
            else:
                output.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(command, check=True)
                print({"task": task["out_id"], "reference": kind, "status": "complete"}, flush=True)


if __name__ == "__main__":
    main()
