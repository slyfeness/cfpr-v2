"""Deterministic, pixel-level counterfactual edits for evaluator unit tests.

The module intentionally contains no evaluator thresholds or model calls.  Each
edit returns both its result and a machine-checkable description of what it
changed, so the runner can reject an invalid intervention before scoring it.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.signal import correlate, correlation_lags


def _window(n: int, start: int, end: int) -> range:
    if not (0 <= start <= end <= n):
        raise ValueError(f"invalid half-open window [{start}, {end}) for {n} frames")
    return range(start, end)


def changed_frame_indices(before: np.ndarray, after: np.ndarray) -> list[int]:
    if before.shape != after.shape or before.dtype != after.dtype:
        raise AssertionError("edit changed array shape or dtype")
    return np.flatnonzero(np.any(before != after, axis=tuple(range(1, before.ndim)))).astype(int).tolist()


def assert_changed_frames(before: np.ndarray, after: np.ndarray, expected: Iterable[int]) -> None:
    actual = changed_frame_indices(before, after)
    wanted = sorted(set(int(x) for x in expected))
    if actual != wanted:
        raise AssertionError(f"changed frames {actual}, expected {wanted}")


def replace_window(clean: np.ndarray, donor: np.ndarray, start: int, end: int):
    if clean.shape != donor.shape:
        raise ValueError("clean and donor must have the same shape")
    out = clean.copy()
    out[start:end] = donor[start:end]
    return out, {"kind": "identity_swap", "window": [start, end],
                 "source_frame_map": {str(i): i for i in _window(len(clean), start, end)}}


def freeze_window(clean: np.ndarray, anchor: int, length: int = 12):
    end = anchor + length
    _window(len(clean), anchor, end)
    out = clean.copy()
    out[anchor:end] = clean[anchor]
    return out, {"kind": "freeze", "anchor": anchor, "window": [anchor, end],
                 "source_frame_map": {str(i): anchor for i in _window(len(clean), anchor, end)}}


def skip_forward_window(clean: np.ndarray, start: int, end: int, shift: int = 6):
    _window(len(clean), start, end)
    if shift <= 0:
        raise ValueError("shift must be positive")
    out = clean.copy()
    mapping = {}
    for i in _window(len(clean), start, end):
        source = min(i + shift, len(clean) - 1)
        out[i] = clean[source]
        mapping[str(i)] = source
    return out, {"kind": "skip_forward", "window": [start, end], "shift": shift,
                 "source_frame_map": mapping}


def jump_frame(clean: np.ndarray, frame: int, jump: int = 30):
    _window(len(clean), frame, frame + 1)
    source = min(frame + jump, len(clean) - 1)
    out = clean.copy()
    out[frame] = clean[source]
    return out, {"kind": "motion_jump", "frame": frame, "jump": jump,
                 "source_frame_map": {str(frame): source}}


def _srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    x = rgb.astype(np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    x = np.clip(linear, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * (x ** (1 / 2.4)) - 0.055)
    return np.rint(np.clip(srgb * 255.0, 0, 255)).astype(np.uint8)


def apply_flicker_ev(clean: np.ndarray, start: int, end: int, ev: float):
    _window(len(clean), start, end)
    if ev <= 0:
        raise ValueError("EV must be positive")
    out = clean.copy()
    linear = _srgb_to_linear(clean[start:end])
    factors = np.array([2 ** (ev / 2) if i % 2 == 0 else 2 ** (-ev / 2)
                        for i in _window(len(clean), start, end)], dtype=np.float64)
    scaled = linear * factors[:, None, None, None]
    out[start:end] = _linear_to_srgb(scaled)
    return out, {"kind": "flicker", "window": [start, end], "ev": float(ev),
                 "factors": factors.tolist(), "clipped_fraction": float(np.mean((scaled < 0) | (scaled > 1)))}


def hard_cut_window(clean: np.ndarray, start: int, end: int, offset: int = 48):
    _window(len(clean), start, end)
    if offset <= 0:
        raise ValueError("offset must be positive")
    out = clean.copy()
    mapping = {}
    for i in _window(len(clean), start, end):
        source = min(i + offset, len(clean) - 1)
        out[i] = clean[source]
        mapping[str(i)] = source
    return out, {"kind": "hard_cut", "window": [start, end], "offset": offset,
                 "source_frame_map": mapping}


def _pixel_points(points: np.ndarray, width: int, height: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if np.nanmax(np.abs(pts)) <= 1.5:
        pts[:, 0] *= width
        pts[:, 1] *= height
    return pts


def occlude_right_hand(clean: np.ndarray, hands: np.ndarray, hand_scores: np.ndarray, start: int, end: int,
                       *, right_hand_index: int, margin: float = 0.20):
    _window(len(clean), start, end)
    if hands.shape[:3] != hand_scores.shape:
        raise ValueError("hands and hand_scores shapes do not agree")
    height, width = clean.shape[1:3]
    out = clean.copy()
    boxes, applied, skipped = {}, [], []
    for frame in _window(len(clean), start, end):
        points = np.asarray(hands[frame, right_hand_index], dtype=float)
        valid = (np.asarray(hand_scores[frame, right_hand_index], dtype=float) >= 0.3) & np.isfinite(points).all(axis=1)
        if not np.any(valid):
            skipped.append(frame)
            continue
        points = _pixel_points(points[valid], width, height)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        dx, dy = max(1.0, x1 - x0), max(1.0, y1 - y0)
        x0 = max(0, int(np.floor(x0 - margin * dx)))
        y0 = max(0, int(np.floor(y0 - margin * dy)))
        x1 = min(width - 1, int(np.ceil(x1 + margin * dx)))
        y1 = min(height - 1, int(np.ceil(y1 + margin * dy)))
        out[frame, y0:y1 + 1, x0:x1 + 1] = 128
        boxes[str(frame)] = [x0, y0, x1, y1]
        applied.append(frame)
    return out, {"kind": "right_hand_occlusion", "window": [start, end], "right_hand_index": right_hand_index,
                 "margin": margin, "boxes": boxes, "applied_frames": applied, "skipped_frames": skipped}


def shift_audio_window(audio: np.ndarray, start: int, end: int, shift_samples: int):
    if audio.ndim != 1:
        raise ValueError("audio must be mono")
    _window(len(audio), start, end)
    if shift_samples == 0 or abs(shift_samples) >= end - start:
        raise ValueError("shift must be nonzero and smaller than the window")
    out = audio.copy()
    n = abs(int(shift_samples))
    if shift_samples > 0:
        out[start:start + n] = 0
        out[start + n:end] = audio[start:end - n]
    else:
        out[end - n:end] = 0
        out[start:end - n] = audio[start + n:end]
    return out, {"kind": "audio_shift", "sample_window": [start, end], "shift_samples": int(shift_samples)}


def verify_audio_shift_xcorr(original: np.ndarray, shifted: np.ndarray, start: int, end: int,
                             expected_shift_samples: int, tolerance_samples: int = 1) -> dict:
    """Verify a window-local signed audio delay by FFT cross-correlation.

    The returned lag follows ``shifted(t) = original(t - lag)``. A positive
    requested shift therefore delays the audio and has a positive recovered lag.
    """
    if original.ndim != 1 or shifted.ndim != 1 or original.shape != shifted.shape:
        raise ValueError("audio arrays must be equal-length mono vectors")
    _window(len(original), start, end)
    x = original[start:end].astype(np.float64)
    y = shifted[start:end].astype(np.float64)
    x -= x.mean()
    y -= y.mean()
    corr = correlate(y, x, mode="full", method="fft")
    lag = int(correlation_lags(len(y), len(x), mode="full")[int(np.argmax(corr))])
    valid = abs(lag - int(expected_shift_samples)) <= int(tolerance_samples)
    if not valid:
        raise AssertionError(
            f"cross-correlation lag {lag}, expected {expected_shift_samples} +/- {tolerance_samples}"
        )
    return {
        "method": "fft_cross_correlation",
        "recovered_lag_samples": lag,
        "expected_shift_samples": int(expected_shift_samples),
        "tolerance_samples": int(tolerance_samples),
        "passed": True,
    }


def align_emotion_scores(scores: np.ndarray, start_seconds: float, n_frames: int, *, src_fps: float = 30.0,
                         dst_fps: float = 24.0):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 2:
        raise ValueError("emotion scores must be [frames, classes]")
    indices = (float(start_seconds) + np.arange(n_frames, dtype=float) / dst_fps) * src_fps
    grid = np.arange(len(values), dtype=float)
    clipped = np.clip(indices, 0, len(values) - 1)
    aligned = np.stack([np.interp(clipped, grid, values[:, c]) for c in range(values.shape[1])], axis=1)
    return aligned, indices
