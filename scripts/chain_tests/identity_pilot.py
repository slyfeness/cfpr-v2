"""Frozen P0--P7 context-only identity injection primitives.

All edits are consumed through ACC's existing ``ctx_edit`` callback.  This
module deliberately has no dependency on the generator core or evaluator.
"""
from __future__ import annotations

import numpy as np
import torch

from scripts.chain_tests.identity_edit import context_weight, face_token_mask


WINDOW = (96, 144)


def pilot_specs():
    """Return the task-book's exhaustive, ordered P0--P7 registry."""
    return [
        {"id": "P0", "name": "whole_bank_raw_donor", "window": list(WINDOW)},
        {"id": "P1", "name": "face_replace", "window": list(WINDOW)},
        {"id": "P2", "name": "face_replace_high_resolution", "window": list(WINDOW)},
        {"id": "P3", "name": "face_replace_low_resolution", "window": list(WINDOW)},
        {"id": "P4", "name": "face_replace_random_half", "window": list(WINDOW)},
        {"id": "P5", "name": "append_donor_face", "window": list(WINDOW)},
        {"id": "P6", "name": "remove_own_then_append_donor_face", "window": list(WINDOW)},
        {"id": "P7", "name": "whole_bank_pixel_swap_reference", "window": list(WINDOW)},
    ]


def _blend(own: torch.Tensor, donor: torch.Tensor, mask: torch.Tensor, weight: float) -> torch.Tensor:
    if weight == 0:
        return own
    result = own.clone()
    result[:, mask] = ((1.0 - weight) * own[:, mask].float() + weight * donor[:, mask].float()).to(own.dtype)
    return result


def _random_half(mask: torch.Tensor, layer_index: int) -> torch.Tensor:
    selected = torch.zeros_like(mask)
    choices = torch.where(mask)[0].detach().cpu().numpy()
    count = len(choices) // 2
    if count:
        picked = np.random.default_rng(layer_index).choice(choices, size=count, replace=False)
        selected[torch.as_tensor(picked, device=mask.device)] = True
    return selected


def make_pilot_ctx_edit(readers, aligned_donor_bank, full_replace_bank, own_box, action):
    """Build the frozen P0--P7 intervention and a complete context audit trail.

    ``aligned_donor_bank`` is used by P1--P6. ``full_replace_bank`` is the
    unaligned original donor bank for P0 and the P7 pixel-swap reference bank
    for P7. The caller supplies no other intervention route.
    """
    variant = action["id"]
    if variant not in {spec["id"] for spec in pilot_specs()}:
        raise ValueError(f"unregistered pilot variant: {variant}")
    window = tuple(action["window"])
    state = {"base": None, "face_masks": None, "active_masks": None, "contexts": [], "layer_token_counts": None}

    def initialise():
        if state["base"] is not None:
            return
        if not (len(readers) == len(aligned_donor_bank) == len(full_replace_bank)):
            raise ValueError("reader and donor banks must have equal layer counts")
        base = [reader.bank[0] for reader in readers]
        if any(own.shape != donor.shape for own, donor in zip(base, aligned_donor_bank)):
            raise ValueError("aligned donor bank shape mismatch")
        if any(own.shape != donor.shape for own, donor in zip(base, full_replace_bank)):
            raise ValueError("full replacement bank shape mismatch")
        face = [face_token_mask(own, own_box) for own in base]
        counts = [int(own.shape[1]) for own in base]
        if variant == "P2":
            extreme = max(counts)
            active = [mask if count == extreme else torch.zeros_like(mask) for mask, count in zip(face, counts)]
        elif variant == "P3":
            extreme = min(counts)
            active = [mask if count == extreme else torch.zeros_like(mask) for mask, count in zip(face, counts)]
        elif variant == "P4":
            active = [_random_half(mask, index) for index, mask in enumerate(face)]
        else:
            active = face
        state.update(base=base, face_masks=face, active_masks=active, layer_token_counts=counts)

    def ctx_edit(t_i, frames, _reader):
        initialise()
        w = context_weight(frames, window)
        record = {"timestep_index": int(t_i), "frames": [int(frame) for frame in frames], "w": float(w), "variant": variant}
        if w == 0:
            for reader, own in zip(readers, state["base"]):
                reader.bank = [own]
            record["operation"] = "restore_original"
        elif variant in {"P0", "P7"}:
            for reader, own, donor in zip(readers, state["base"], full_replace_bank):
                reader.bank = [_blend(own, donor.to(own.device), torch.ones(own.shape[1], device=own.device, dtype=torch.bool), w)]
            record["operation"] = "replace_entire_bank" if w == 1 else "blend_entire_bank_transition"
        elif variant in {"P1", "P2", "P3", "P4"}:
            for reader, own, donor, mask in zip(readers, state["base"], aligned_donor_bank, state["active_masks"]):
                reader.bank = [_blend(own, donor.to(own.device), mask, w)]
            record["operation"] = "replace_face" if w == 1 else "blend_face_transition"
            record["selected_face_token_counts"] = [int(mask.sum().item()) for mask in state["active_masks"]]
        elif variant == "P5":
            if w < .5:
                for reader, own in zip(readers, state["base"]):
                    reader.bank = [own]
                record["operation"] = "restore_original_below_concat_threshold"
            else:
                for reader, own, donor, mask in zip(readers, state["base"], aligned_donor_bank, state["face_masks"]):
                    reader.bank = [own, donor.to(own.device)[:, mask]]
                record["operation"] = "append_donor_face"
        else:  # P6
            if w < .5:
                for reader, own in zip(readers, state["base"]):
                    reader.bank = [own]
                record["operation"] = "restore_original_below_concat_threshold"
            else:
                for reader, own, donor, mask in zip(readers, state["base"], aligned_donor_bank, state["face_masks"]):
                    reader.bank = [own[:, ~mask], donor.to(own.device)[:, mask]]
                record["operation"] = "remove_own_append_donor_face"
        state["contexts"].append(record)

    return ctx_edit, state
