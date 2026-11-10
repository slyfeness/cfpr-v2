"""Frozen round-2 identity injection edits: discrete contextual replacement only."""
from __future__ import annotations

import numpy as np
import torch

from scripts.chain_tests.identity_edit import context_weight, face_token_mask

WINDOW = (96, 144)


def round2_specs():
    return [
        {"id": "P1b", "p": 1.0, "pad": .25, "window": list(WINDOW)},
        {"id": "D75", "p": .75, "pad": .25, "window": list(WINDOW)},
        {"id": "D90", "p": .90, "pad": .25, "window": list(WINDOW)},
        {"id": "H50", "p": 1.0, "pad": .50, "window": list(WINDOW)},
    ]


def expand_box(box, pad):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * pad, (y1 - y0) * pad
    return (max(0., x0 - dx), max(0., y0 - dy), min(1., x1 + dx), min(1., y1 + dy))


def _subset(mask, p, layer):
    if p == 1.0:
        return mask
    out = torch.zeros_like(mask)
    choices = torch.where(mask)[0].detach().cpu().numpy()
    n = int(round(len(choices) * p))
    if n:
        picked = np.random.default_rng(layer).choice(choices, n, replace=False)
        out[torch.as_tensor(picked, device=mask.device)] = True
    return out


def make_round2_ctx_edit(readers, donor_bank, box, action):
    """P1b/D75/D90/H50: at w>=.5 direct token replacement, otherwise restore."""
    if action["id"] not in {item["id"] for item in round2_specs()}:
        raise ValueError(f"unregistered round2 variant {action['id']}")
    state = {"base": None, "selected_masks": None, "contexts": [], "selected_face_token_counts": None}
    window = tuple(action["window"])

    def edit(t_i, frames, _reader):
        if state["base"] is None:
            state["base"] = [reader.bank[0] for reader in readers]
            if len(state["base"]) != len(donor_bank) or any(a.shape != b.shape for a, b in zip(state["base"], donor_bank)):
                raise ValueError("round2 bank shape mismatch")
            face = [face_token_mask(own, box) for own in state["base"]]
            state["selected_masks"] = [_subset(mask, float(action["p"]), i) for i, mask in enumerate(face)]
            state["selected_face_token_counts"] = [int(mask.sum().item()) for mask in state["selected_masks"]]
        w = context_weight(frames, window)
        record = {"timestep_index": int(t_i), "frames": [int(x) for x in frames], "w": float(w), "variant": action["id"]}
        if w < .5:
            for reader, own in zip(readers, state["base"]):
                reader.bank = [own]
            record["operation"] = "restore_original_below_threshold"
        else:
            for reader, own, donor, mask in zip(readers, state["base"], donor_bank, state["selected_masks"]):
                value = own.clone(); value[:, mask] = donor.to(own.device)[:, mask]; reader.bank = [value]
            record["operation"] = "direct_replace_at_or_above_threshold"
            record["selected_face_token_counts"] = state["selected_face_token_counts"]
        state["contexts"].append(record)
    return edit, state
