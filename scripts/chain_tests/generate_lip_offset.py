#!/usr/bin/env python3
"""Direct, shared-GPU-safe generator-side audio-feature lip-offset runner.

Each invocation loads ACC once, then processes its selected registered tasks
sequentially.  It intentionally does not use acc_worker or start_workers.sh.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from scripts.chain_tests.audio_edit import apply_audio_shift
from scripts.chain_tests.lip_offset_config import build_registered_tasks
from src import acc_core


ROOT = Path("/data/lanyi/projects/cfpr-v2")
OUT_ROOT = Path("/data/lanyi/outputs/cfpr-v2/chain_lip_offset")
INPUT_ROOT = Path("/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1")


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


class RecordedAudioShift:
    def __init__(self, action):
        self.action = action
        self.meta = None

    def __call__(self, audio):
        window = tuple(self.action["window"])
        k = int(self.action["k"])
        edited = apply_audio_shift(audio, window=window, k=k)
        start, end = window
        exterior = torch.cat((audio[:, :start], audio[:, end:]), dim=1)
        edited_exterior = torch.cat((edited[:, :start], edited[:, end:]), dim=1)
        if k == 0:
            mapping_ok = edited is audio
        else:
            mapping_ok = torch.equal(edited[:, start:end], audio[:, start - k:end - k])
        self.meta = {
            "action": self.action,
            "feature_shape": list(audio.shape),
            "original_sha256": tensor_sha256(audio),
            "edited_sha256": tensor_sha256(edited),
            "outside_window_elementwise_identical": bool(torch.equal(exterior, edited_exterior)),
            "inside_window_source_mapping_verified": bool(mapping_ok),
        }
        if not self.meta["outside_window_elementwise_identical"] or not self.meta["inside_window_source_mapping_verified"]:
            raise AssertionError("registered audio-feature edit invariant failed")
        return edited


def task_state_record(task):
    action = task["actions"][0] if task["actions"] else None
    window = tuple(action["window"]) if action else None
    k = int(action["k"]) if action else 0
    segments = []
    for segment in range(20):
        start, end = segment * 12, (segment + 1) * 12
        intersects = window is not None and start < window[1] and end > window[0]
        segments.append({
            "segment_index": segment,
            "source_frames_24fps": [start, end],
            "planned_audio_shift_24fps": k if intersects else 0,
            "window": list(window) if intersects else None,
            "measured_lipsync_offset_25fps": None,
            "measured_lipsync_confidence": None,
            "seed": task["seed"],
        })
    return {"task": task, "segments": segments}


def make_audio_edit(task):
    if not task["actions"]:
        return None, {"actions": [], "audio_edit_called": False}
    if len(task["actions"]) != 1 or task["actions"][0]["type"] != "audio_shift":
        raise ValueError(f"unsupported actions: {task['actions']}")
    hook = RecordedAudioShift(task["actions"][0])
    return hook, hook


def run_task(pipe, task, output_root):
    destination = output_root / task["out_id"]
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite {destination}")
    reference, audio_path, poses = acc_core.load_inputs(INPUT_ROOT / task["sample"], task["frames"])
    edit, audit = make_audio_edit(task)
    torch.cuda.reset_peak_memory_stats()
    video, info, step_latents = acc_core.generate(
        pipe, reference, audio_path, poses, task["frames"], seed=task["seed"], audio_edit=edit,
    )
    info["chain_lip_offset_task"] = task
    acc_core.save(destination, video, info, audio_path, final_latents=step_latents[-1])
    edit_meta = audit.meta if isinstance(audit, RecordedAudioShift) else audit
    (destination / "edit_meta.json").write_text(json.dumps(edit_meta, ensure_ascii=False, indent=2))
    (destination / "state_record.json").write_text(json.dumps(task_state_record(task), ensure_ascii=False, indent=2))
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "scripts/chain_tests/config.json")
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--kinds", nargs="*")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    tasks = build_registered_tasks(config["samples"], config["windows"], seed=config["seed"], frames=config["frames"])
    if args.samples:
        tasks = [task for task in tasks if task["sample"] in args.samples]
    if args.kinds:
        tasks = [task for task in tasks if task["kind"] in args.kinds]
    if not tasks:
        raise ValueError("no registered tasks selected")
    pipe = acc_core.load_pipe()
    try:
        for index, task in enumerate(tasks, start=1):
            destination = run_task(pipe, task, args.out_root)
            print(json.dumps({"completed": index, "total": len(tasks), "task": task["out_id"], "out_dir": str(destination)}))
    finally:
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
