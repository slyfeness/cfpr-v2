"""Auditable primitives for the registered generator-side identity-drift edit.

The returned callback is consumed only by ACC's pre-existing ``ctx_edit`` hook;
this module never changes the generator core.
"""
from __future__ import annotations

import numpy as np
import torch


def expanded_face_box(points: np.ndarray, scores: np.ndarray, threshold: float = 0.3, pad: float = 0.25):
    valid = np.asarray(scores).reshape(-1) >= threshold
    xy = np.asarray(points)[valid]
    if not len(xy):
        raise ValueError("no confident facial keypoints")
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    dx, dy = pad * (x1 - x0), pad * (y1 - y0)
    return (max(0.0, float(x0 - dx)), max(0.0, float(y0 - dy)), min(1.0, float(x1 + dx)), min(1.0, float(y1 + dy)))


def context_weight(frames, window) -> float:
    start, end = window
    return sum(start <= frame < end for frame in frames) / len(frames)


def mix_tokens(own: torch.Tensor, donor: torch.Tensor, mask: torch.Tensor, strength: float) -> torch.Tensor:
    if strength == 0:
        return own
    if own.shape != donor.shape:
        raise ValueError(f"bank shape mismatch: {tuple(own.shape)} vs {tuple(donor.shape)}")
    out = own.clone()
    out[:, mask] = ((1 - strength) * own[:, mask].float() + strength * donor[:, mask].float()).to(own.dtype)
    return out


def face_token_mask(tokens: torch.Tensor, box) -> torch.Tensor:
    """Select spatial token centres inside a normalized ``(x0,y0,x1,y1)`` box."""
    n = tokens.shape[1]
    side = int(round(n ** 0.5))
    if side * side != n:
        raise ValueError(f"expected square spatial token grid, received {n} tokens")
    ys, xs = torch.meshgrid(
        torch.arange(side, device=tokens.device),
        torch.arange(side, device=tokens.device),
        indexing="ij",
    )
    cx, cy = (xs.flatten() + 0.5) / side, (ys.flatten() + 0.5) / side
    x0, y0, x1, y1 = box
    return (cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1)


def make_identity_ctx_edit(readers, donor_bank, own_box, action):
    """Return a context callback plus auditable metadata state.

    For zero effective strength it restores the exact pristine own-bank object,
    rather than a numerical copy.  All non-zero arithmetic is in float32.
    """
    alpha = float(action["alpha"])
    window = tuple(action["window"])
    state = {
        "base": None,
        "masks": None,
        "token_counts": None,
        "step0_contexts": [],
        "checks": {
            "zero_strength_context_count": 0,
            "zero_strength_elementwise": True,
            "zero_strength_original_object": True,
            "outside_face_identical": True,
            "inside_formula_max_abs_error": 0.0,
            "inside_formula_within_fp16_rounding": True,
        },
    }

    def ctx_edit(t_i, frames, _reader):
        if state["base"] is None:
            if len(readers) != len(donor_bank):
                raise ValueError(f"reader/donor bank layer mismatch: {len(readers)} vs {len(donor_bank)}")
            # References to reader.update's pristine own-bank objects.  We only
            # ever replace reader.bank; the objects stored here stay untouched.
            state["base"] = [reader.bank[0] for reader in readers]
            state["masks"] = [face_token_mask(own, own_box) for own in state["base"]]
            state["token_counts"] = [int(mask.sum().item()) for mask in state["masks"]]

        weight = context_weight(frames, window)
        strength = alpha * weight
        for reader, own, donor, mask in zip(readers, state["base"], donor_bank, state["masks"]):
            mixed = mix_tokens(own, donor.to(own.device), mask, strength)
            reader.bank = [mixed]
            actual = reader.bank[0]
            if strength == 0:
                state["checks"]["zero_strength_context_count"] += 1
                state["checks"]["zero_strength_original_object"] &= actual is own
                state["checks"]["zero_strength_elementwise"] &= bool(torch.equal(actual, own))
            else:
                expected = ((1 - strength) * own[:, mask].float() + strength * donor.to(own.device)[:, mask].float()).to(own.dtype)
                error = float((actual[:, mask].float() - expected.float()).abs().max().item()) if bool(mask.any()) else 0.0
                state["checks"]["inside_formula_max_abs_error"] = max(state["checks"]["inside_formula_max_abs_error"], error)
                state["checks"]["inside_formula_within_fp16_rounding"] &= error <= 2 ** -10
                state["checks"]["outside_face_identical"] &= bool(torch.equal(actual[:, ~mask], own[:, ~mask]))
        if t_i == 0:
            state["step0_contexts"].append(
                {"start_frame": int(frames[0]), "frames": [int(frame) for frame in frames], "w": float(weight), "effective_strength": float(strength)}
            )

    return ctx_edit, state


def identity_state_record(task, step0_contexts, segment_count: int = 20):
    """Format the task's 12-frame planned dose trace without reading a model."""
    action = task["actions"][0] if task["actions"] else None
    donor = action["donor"] if action else None
    per_frame = {}
    for context in step0_contexts:
        for frame in context["frames"]:
            per_frame.setdefault(frame, []).append(context["effective_strength"])
    segments = []
    for index in range(segment_count):
        start, end = 12 * index, 12 * (index + 1)
        strength = sum(sum(per_frame.get(frame, [0.0])) / len(per_frame.get(frame, [0.0])) for frame in range(start, end)) / 12
        segments.append({
            "segment_index": index,
            "source_frames_24fps": [start, end],
            "planned_identity_mix_strength": strength,
            "measured_arcface_own": None,
            "measured_arcface_donor": None,
            "valid_face_frames": None,
            "seed": task["seed"],
            "donor": donor,
        })
    return {"task": task, "segments": segments}
