#!/usr/bin/env python3
"""Create only missing identity-drift state records from existing edit metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.chain_tests.identity_config import build_tasks
from scripts.chain_tests.identity_edit import identity_state_record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    repaired = []
    for task in build_tasks():
        directory = args.root / task["out_id"]
        summary, meta, record = directory / "summary.json", directory / "edit_meta.json", directory / "state_record.json"
        if not summary.exists() or not meta.exists() or record.exists():
            continue
        payload = json.loads(meta.read_text())
        record.write_text(json.dumps(identity_state_record(task, payload["step0_contexts"]), ensure_ascii=False, indent=2))
        repaired.append(task["out_id"])
    print(json.dumps({"created_missing_state_records": repaired}))


if __name__ == "__main__":
    main()
