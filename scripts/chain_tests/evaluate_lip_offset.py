#!/usr/bin/env python3
"""Run the registered SyncNet measurement for generated lip-offset videos."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from scripts.chain_tests.lip_offset_config import build_registered_tasks


INPUT_ROOT = Path("/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1")
DEFAULT_EVALUATOR = Path("scripts/evaluators/eval_lipsync.py")
DEFAULT_PYTHON = Path("/data/lanyi/.conda/envs/syncnet/bin/python")


def lipsync_command(root: Path, task: dict, evaluator: Path, python: Path) -> list[str]:
    task_root = root / task["out_id"]
    return [
        str(python),
        str(evaluator),
        "gen",
        str(task_root / "frames_uint8.npy"),
        str(INPUT_ROOT / task["sample"] / "audio.wav"),
        str(task_root / "evaluators/lipsync"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("scripts/chain_tests/config.json"))
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    tasks = build_registered_tasks(config["samples"], config["windows"], seed=config["seed"], frames=config["frames"])
    if args.samples:
        tasks = [task for task in tasks if task["sample"] in args.samples]
    for task in tasks:
        command = lipsync_command(args.root, task, DEFAULT_EVALUATOR, DEFAULT_PYTHON)
        output = Path(command[-1]) / "gen_lipsync.json"
        if output.exists():
            print(json.dumps({"task": task["out_id"], "status": "already_present"}))
            continue
        if args.dry_run:
            print(json.dumps({"task": task["out_id"], "command": command}))
            continue
        Path(command[-1]).mkdir(parents=True, exist_ok=False)
        subprocess.run(command, check=True)
        print(json.dumps({"task": task["out_id"], "status": "complete"}))


if __name__ == "__main__":
    main()
