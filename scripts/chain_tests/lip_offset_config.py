"""Pre-registered matrix construction for the generator-side lip-offset test."""

from __future__ import annotations


def build_registered_tasks(samples, windows, *, seed, frames=240):
    """Return the fixed 15-task matrix for each sample without sampling choices."""
    tasks = []
    for sample in samples:
        for replica in ("a", "b"):
            tasks.append({
                "sample": sample, "frames": frames, "seed": seed,
                "actions": [], "kind": f"clean_{replica}",
                "out_id": f"{sample}/clean_{replica}",
            })
        tasks.append({
            "sample": sample, "frames": frames, "seed": seed,
            "actions": [{"type": "audio_shift", "window": windows["W1"], "k": 0}],
            "kind": "zero_action", "out_id": f"{sample}/zero_action",
        })
        for window_name, window in windows.items():
            for k in (-8, -4, -2, 2, 4, 8):
                tasks.append({
                    "sample": sample, "frames": frames, "seed": seed,
                    "actions": [{"type": "audio_shift", "window": window, "k": k}],
                    "kind": f"{window_name}_k_{k:+d}", "window": window_name,
                    "out_id": f"{sample}/{window_name}/k_{k:+d}",
                })
    return tasks
