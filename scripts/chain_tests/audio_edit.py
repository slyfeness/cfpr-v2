"""Audio-feature edits for the generator-side lip-offset chain test."""

from __future__ import annotations

import torch


def apply_audio_shift(audio: torch.Tensor, *, window: tuple[int, int], k: int) -> torch.Tensor:
    """Shift feature blocks in ``window`` by ``k`` without changing exterior blocks.

    For output frame ``t`` in ``[a, b)``, the new block is the original
    block at ``t-k``.  k=0 deliberately returns the exact input object.
    """
    if audio.ndim < 2:
        raise ValueError("audio must have a frame dimension at axis 1")
    start, end = window
    length = audio.shape[1]
    if not (0 <= start <= end <= length):
        raise ValueError(f"window {window} outside audio length {length}")
    if k == 0:
        return audio
    source_start, source_end = start - k, end - k
    if not (0 <= source_start and source_end <= length):
        raise ValueError(f"audio-shift source [{source_start}, {source_end}) outside feature length {length}")
    edited = audio.clone()
    edited[:, start:end] = audio[:, source_start:source_end]
    return edited


def audio_shift_edit(window: tuple[int, int], k: int):
    """Build the optional ``acc_core.generate(audio_edit=...)`` hook."""
    return lambda audio: apply_audio_shift(audio, window=window, k=k)
