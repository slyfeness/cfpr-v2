#!/usr/bin/env python3
"""Corrected post-hoc D1/D2 and new D6 diagnostics.

This script reads frozen evaluator outputs only.  It neither invokes nor
modifies an evaluator, and writes a new post-hoc JSON artifact and diagnostic
figures only after all SyncNet-curve reconstructions match their stored JSON.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile


ROOT = Path("/data/lanyi/projects/cfpr-v2")
OUT = Path("/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2")
CFG = json.loads((ROOT / "scripts/evaluator_unit_tests/config.json").read_text())
SAMPLES = CFG["samples"]
WINDOWS = CFG["windows"]
LIPSYNC_SHIFT = 15
DIAG_JSON = OUT / "posthoc_diagnosis_fix_20261109.json"
MEDIA = ROOT / "docs/status/evaluator_unit_tests_20261108_media/posthoc_diagnosis_fix_20261109"


def json_load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def lipsync_json(directory):
    return json_load(Path(directory) / "gen_lipsync.json")


def lipsync_dir(sample, test=None, window=None, level=None):
    if test is None:
        return OUT / "clean" / sample / "evaluators/lipsync"
    return OUT / test / sample / window / level / "evaluators/lipsync"


def segment_mask(start, end, count=20):
    return np.asarray([(index * 12 < end and (index + 1) * 12 > start) for index in range(count)])


def finite_median(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else float("nan")


def classify_d1_window(measurements):
    """Apply the corrected D1 priority: inaccurate before undetermined."""
    if any(item["valid_segments"] >= 2 and item["abs_error"] > 1 for item in measurements):
        return "inaccurate"
    if any(item["valid_segments"] < 2 for item in measurements):
        return "undetermined"
    return "accurate"


def classify_d1_person(window_statuses):
    if any(status == "inaccurate" for status in window_statuses):
        return "inaccurate"
    if all(status == "accurate" for status in window_statuses):
        return "accurate"
    return "undetermined"


def d1_measurement(clean, edited, active_segments, threshold, k):
    clean_offsets = np.asarray(clean["seg_offsets"], dtype=float)
    clean_conf = np.asarray(clean["seg_conf"], dtype=float)
    delta = np.asarray(edited["seg_offsets"], dtype=float) - clean_offsets
    valid = active_segments & (clean_conf >= threshold) & np.isfinite(delta)
    count = int(valid.sum())
    median = finite_median(delta[valid])
    expected = -k * 25.0 / 24.0
    error = abs(median - expected) if np.isfinite(median) else float("nan")
    return {
        "k_24fps": k,
        "valid_segments": count,
        "median_delta_25fps": median,
        "expected_delta_25fps": expected,
        "abs_error": error,
        "relative_accurate": bool(count >= 2 and error <= 1),
        "direction_correct": bool(count >= 2 and median * expected > 0),
    }


def d1_corrected():
    rows = []
    summaries = {}
    stratified_rows = []
    for threshold in (2, 3, 4):
        people = {}
        for sample in SAMPLES:
            clean = lipsync_json(lipsync_dir(sample))
            window_results = []
            for window, (start, end) in WINDOWS.items():
                mask = segment_mask(start, end)
                measures = []
                for k in (-8, -4, -2, 2, 4, 8):
                    edited = lipsync_json(lipsync_dir(sample, "T1", window, f"k_{k:+d}"))
                    measure = d1_measurement(clean, edited, mask, threshold, k)
                    rows.append({
                        "sample": sample,
                        "window": window,
                        "confidence_threshold": threshold,
                        **measure,
                    })
                    if abs(k) in (4, 8):
                        measures.append(measure)
                    if threshold == 3 and sample in SAMPLES[1:] and measure["valid_segments"] >= 2:
                        stratified_rows.append({"sample": sample, "window": window, **measure})
                window_results.append({"window": window, "status": classify_d1_window(measures), "measurements": measures})
            people[sample] = {
                "status": classify_d1_person([result["status"] for result in window_results]),
                "windows": window_results,
            }
        unseen = people.copy()
        unseen.pop(SAMPLES[0])
        undetermined = [sample for sample, value in unseen.items() if value["status"] == "undetermined"]
        summaries[str(threshold)] = {
            "people": people,
            "other_seven_accurate": sum(value["status"] == "accurate" for value in unseen.values()),
            "other_seven_determinable": sum(value["status"] != "undetermined" for value in unseen.values()),
            "other_seven_undetermined": undetermined,
            "determinable_at_least_five": sum(value["status"] != "undetermined" for value in unseen.values()) >= 5,
        }
    strata = []
    for magnitude in (2, 4, 8):
        selected = [row for row in stratified_rows if abs(row["k_24fps"]) == magnitude]
        denominator = len(selected)
        strata.append({
            "abs_k_24fps": magnitude,
            "determinable_window_direction_count": denominator,
            "relative_accurate_count": sum(row["relative_accurate"] for row in selected),
            "relative_accurate_fraction": (sum(row["relative_accurate"] for row in selected) / denominator if denominator else None),
            "direction_correct_count": sum(row["direction_correct"] for row in selected),
            "direction_correct_fraction": (sum(row["direction_correct"] for row in selected) / denominator if denominator else None),
        })
    return {"rows": rows, "summaries": summaries, "threshold3_stratified": strata}


def strict_smooth(values, width=3):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    sums = np.convolve(np.where(finite, values, 0.0), np.ones(width), mode="same")
    counts = np.convolve(finite.astype(int), np.ones(width), mode="same")
    return np.where(counts == width, sums / width, np.nan)


def mouth_amplitude(face, face_score):
    required = (61, 67, 62, 66, 63, 65, 36, 45)
    valid = np.all(np.asarray(face_score)[:, required] >= 0.3, axis=1)
    pairs = ((61, 67), (62, 66), (63, 65))
    opening = np.mean([np.linalg.norm(face[:, upper] - face[:, lower], axis=1) for upper, lower in pairs], axis=0)
    eye_width = np.linalg.norm(face[:, 36] - face[:, 45], axis=1)
    output = opening / np.maximum(eye_width, 1e-6)
    output[~valid] = np.nan
    return strict_smooth(output, width=3)


def audio_envelope(audio, sample_rate, fps):
    audio = np.asarray(audio, dtype=float)
    sos = signal.butter(4, [300, 3000], btype="bandpass", fs=sample_rate, output="sos")
    filtered = signal.sosfiltfilt(sos, audio)
    edges = np.rint(np.arange(int(round(len(audio) / sample_rate * fps)) + 1) * sample_rate / fps).astype(int)
    envelope = np.asarray([
        np.sqrt(np.mean(filtered[edges[index]:edges[index + 1]] ** 2))
        for index in range(len(edges) - 1)
    ])
    return np.convolve(envelope, np.ones(3) / 3, mode="same")


def aligned_at_audio_lag(visual, audio, lag):
    """Pair visual[t] with audio[t + lag]; positive is delayed audio."""
    if lag >= 0:
        return visual[: len(visual) - lag], audio[lag:]
    return visual[-lag:], audio[: len(audio) + lag]


def best_audio_lag(visual, audio, max_lag=10):
    """Return positive-audio-delay lag, correlation, and the full curve."""
    visual = np.asarray(visual, dtype=float)
    audio = np.asarray(audio, dtype=float)
    if len(visual) != len(audio):
        raise ValueError("visual and audio series must have equal frame counts")
    curve = []
    for lag in range(-max_lag, max_lag + 1):
        x, y = aligned_at_audio_lag(visual, audio, lag)
        valid = np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 3:
            correlation = float("nan")
        else:
            x, y = x[valid], y[valid]
            x, y = x - x.mean(), y - y.mean()
            denom = np.linalg.norm(x) * np.linalg.norm(y)
            correlation = float(np.dot(x, y) / denom) if denom else float("nan")
        curve.append({"lag_24fps": lag, "correlation": correlation})
    finite = [entry for entry in curve if np.isfinite(entry["correlation"])]
    if not finite:
        return None, float("nan"), curve
    best = max(finite, key=lambda entry: entry["correlation"])
    return best["lag_24fps"], best["correlation"], curve


def calibrate_audio_lag(envelope):
    """Calibrate sign on one audio envelope before comparing it to mouth motion."""
    base_lag, base_correlation, _ = best_audio_lag(envelope, envelope)
    delayed = np.pad(envelope, (5, 0))[: len(envelope)]
    delayed_lag, delayed_correlation, _ = best_audio_lag(envelope, delayed)
    return {
        "base_lag_24fps": base_lag,
        "base_peak_correlation": base_correlation,
        "audio_delayed_by_24fps_frames": 5,
        "detected_lag_24fps": delayed_lag,
        "detected_change_24fps": (delayed_lag - base_lag if base_lag is not None and delayed_lag is not None else None),
        "delayed_peak_correlation": delayed_correlation,
    }


def clean_face_and_audio(sample):
    data = np.load(OUT / "clean" / sample / "evaluators/dwpose.npz")
    mouth = mouth_amplitude(data["face"], data["face_score"])
    sample_rate, audio = wavfile.read(OUT.parent / "prepared_inputs_aligned_v1" / sample / "audio.wav")
    envelope = audio_envelope(audio, sample_rate, fps=24)
    count = min(len(mouth), len(envelope), CFG["frames"])
    return mouth[:count], envelope[:count]


def save_d2_plot(sample, curve, syncnet_offset, equivalent_offset):
    x = np.asarray([-entry["lag_24fps"] * 25.0 / 24.0 for entry in curve], dtype=float)
    y = np.asarray([entry["correlation"] for entry in curve], dtype=float)
    fig, axis = plt.subplots(figsize=(6.4, 3.6), dpi=140)
    axis.plot(x, y, marker="o", ms=3, lw=1.2)
    axis.axvline(syncnet_offset, color="tab:orange", label=f"SyncNet {syncnet_offset:g}")
    axis.axvline(equivalent_offset, color="tab:green", ls="--", label=f"mouth/audio {equivalent_offset:.2f}")
    axis.set(xlabel="offset (SyncNet sign, 25 fps frames)", ylabel="correlation", title=f"D2′ {sample}")
    axis.legend(fontsize=7)
    axis.grid(alpha=.25)
    fig.tight_layout()
    destination = MEDIA / f"d2_{sample}.png"
    fig.savefig(destination, dpi=140)
    plt.close(fig)
    return str(destination.relative_to(ROOT))


def d2_corrected():
    _, calibration_env = clean_face_and_audio(SAMPLES[0])
    calibration = calibrate_audio_lag(calibration_env)
    if calibration["detected_change_24fps"] != 5:
        raise RuntimeError("D2′ lag calibration did not yield a +5-frame change")
    rows = []
    for sample in SAMPLES:
        mouth, envelope = clean_face_and_audio(sample)
        lag_24, correlation, curve = best_audio_lag(mouth, envelope)
        syncnet = lipsync_json(lipsync_dir(sample))
        syncnet_offset = float(syncnet["av_offset_frames"])
        equivalent = -lag_24 * 25.0 / 24.0 if lag_24 is not None else None
        determinable = lag_24 is not None and abs(lag_24) < 10 and correlation >= .3
        if not determinable:
            conclusion = "undetermined"
            difference = None
        else:
            difference = abs(equivalent - syncnet_offset)
            conclusion = "consistent_more_likely_data_offset" if difference <= 1 else "inconsistent_more_likely_syncnet_bias"
        rows.append({
            "sample": sample,
            "mouth_audio_lag_24fps_positive_audio_delayed": lag_24,
            "peak_correlation": correlation,
            "mouth_audio_equivalent_syncnet_sign_25fps": equivalent,
            "syncnet_global_offset_25fps": syncnet_offset,
            "syncnet_seg_median_offset_25fps": finite_median(syncnet["seg_offsets"]),
            "difference_25fps": difference,
            "determinable": determinable,
            "conclusion": conclusion,
            "curve": curve,
            "plot": save_d2_plot(sample, curve, syncnet_offset, equivalent if equivalent is not None else 0.0),
        })
    return {
        "calibration": {
            "sample": SAMPLES[0],
            **calibration,
            "sign_mapping": "positive mouth/audio lag means audio is delayed; SyncNet offset changes by -k*25/24 for a +k-frame audio delay",
        },
        "rows": rows,
    }


def sync_curve_from_dists(dists):
    curve = np.asarray(dists, dtype=float).mean(axis=1)
    index = int(np.argmin(curve))
    return {
        "offset_25fps": LIPSYNC_SHIFT - index,
        "confidence": float(np.median(curve) - curve[index]),
        "offsets_25fps": [LIPSYNC_SHIFT - index for index in range(len(curve))],
        "distances": curve.tolist(),
    }


def segment_curve_from_dists(dists, segment):
    start = int(round(segment * 12.5))
    end = min(int(round((segment + 1) * 12.5)), dists.shape[1])
    if end - start < 3:
        return None
    return sync_curve_from_dists(dists[:, start:end])


def verify_syncnet_reconstruction(directory):
    saved = lipsync_json(directory)
    dists = np.load(directory / "gen_lipsync.npz")["dists"]
    rebuilt = sync_curve_from_dists(dists)
    if rebuilt["offset_25fps"] != int(saved["av_offset_frames"]) or round(rebuilt["confidence"], 3) != float(saved["confidence"]):
        raise RuntimeError(f"SyncNet global reconstruction mismatch: {directory}")
    for segment, saved_offset in enumerate(saved["seg_offsets"]):
        rebuilt_segment = segment_curve_from_dists(dists, segment)
        if rebuilt_segment is None:
            if saved_offset is not None:
                raise RuntimeError(f"SyncNet segment reconstruction mismatch: {directory}, segment {segment}")
            continue
        if rebuilt_segment["offset_25fps"] != saved_offset or round(rebuilt_segment["confidence"], 3) != float(saved["seg_conf"][segment]):
            raise RuntimeError(f"SyncNet segment reconstruction mismatch: {directory}, segment {segment}")
    return dists, saved


def local_minima_offsets(curve):
    distances = np.asarray(curve["distances"], dtype=float)
    positions = signal.find_peaks(-distances)[0]
    ordered = positions[np.argsort(distances[positions])]
    return [LIPSYNC_SHIFT - int(index) for index in ordered]


def audio_period_25fps(sample, start_24, end_24):
    rate, audio = wavfile.read(OUT.parent / "prepared_inputs_aligned_v1" / sample / "audio.wav")
    envelope = audio_envelope(audio, rate, fps=25)
    start = int(round(start_24 * 25.0 / 24.0))
    end = int(round(end_24 * 25.0 / 24.0))
    values = envelope[start:end]
    values = values - values.mean()
    correlation = np.correlate(values, values, mode="full")[len(values) - 1:]
    correlation = correlation / correlation[0] if correlation[0] else correlation
    candidates = np.arange(2, min(25, len(correlation)))
    peak = int(candidates[np.argmax(correlation[candidates])]) if len(candidates) else None
    return {"period_25fps": peak, "autocorrelation": correlation[:25].tolist()}


def save_d6_plot(label, segments):
    fig, axes = plt.subplots(4, 5, figsize=(15, 10), dpi=120, sharex=True)
    for row, axis in zip(segments, axes.ravel()):
        clean, edited = row["clean_curve"], row["edited_curve"]
        axis.plot(clean["offsets_25fps"], clean["distances"], lw=1, label="clean")
        axis.plot(edited["offsets_25fps"], edited["distances"], lw=1, label="edited")
        axis.axvline(row["expected_edited_offset_25fps"], color="tab:red", ls="--", lw=.8)
        axis.set_title(f"{row['segment_index']}: C {row['clean_offset_25fps']} / E {row['edited_offset_25fps']}", fontsize=7)
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"D6 {label}: dashed = expected edited offset", fontsize=11)
    fig.tight_layout()
    destination = MEDIA / f"d6_{label}.png"
    fig.savefig(destination, dpi=120)
    plt.close(fig)
    return str(destination.relative_to(ROOT))


def d6_conclusion(segments):
    """Use the conservative third category unless direct second-peak evidence exists."""
    actual = [row["edited_offset_25fps"] for row in segments]
    secondary = [row["edited_local_minima_offsets"][1:] for row in segments]
    if any(value in candidates for value, candidates in zip(actual, secondary)):
        return "second_peak_present_and_readout_on_second_peak"
    expected_is_local_minimum = [
        any(abs(candidate - row["expected_edited_offset_25fps"]) <= 1 for candidate in row["edited_local_minima_offsets"])
        for row in segments
    ]
    if not any(expected_is_local_minimum):
        return "expected_position_flat_or_no_clear_peak"
    return "other_or_undetermined"


def d6():
    cases = [
        ("CRB100_025", "W1", 8), ("CRB100_025", "W2", 8),
        ("CRB100_050", "W2", -8), ("CRB100_002", "W1", 8),
        ("CRB100_090", "W1", -8), ("CRB100_090", "W1", 8),
    ]
    loaded = []
    for sample, window, k in cases:
        clean_dir = lipsync_dir(sample)
        edit_dir = lipsync_dir(sample, "T1", window, f"k_{k:+d}")
        clean_dists, clean_json = verify_syncnet_reconstruction(clean_dir)
        edit_dists, edit_json = verify_syncnet_reconstruction(edit_dir)
        loaded.append((sample, window, k, clean_dists, clean_json, edit_dists, edit_json))
    rows = []
    for sample, window, k, clean_dists, clean_json, edit_dists, edit_json in loaded:
        start, end = WINDOWS[window]
        segments = []
        for index in range(20):
            clean_curve = segment_curve_from_dists(clean_dists, index)
            edited_curve = segment_curve_from_dists(edit_dists, index)
            if clean_curve is None or edited_curve is None:
                continue
            expected = clean_curve["offset_25fps"] - k * 25.0 / 24.0
            segments.append({
                "segment_index": index,
                "segment_source_frames_24fps": [index * 12, (index + 1) * 12],
                "in_window": bool(segment_mask(start, end)[index]),
                "clean_offset_25fps": clean_curve["offset_25fps"],
                "clean_confidence": clean_curve["confidence"],
                "edited_offset_25fps": edited_curve["offset_25fps"],
                "edited_confidence": edited_curve["confidence"],
                "expected_edited_offset_25fps": expected,
                "clean_curve": clean_curve,
                "edited_curve": edited_curve,
                "edited_local_minima_offsets": local_minima_offsets(edited_curve),
            })
        in_window = [row for row in segments if row["in_window"]]
        label = f"{sample}_{window}_k_{k:+d}".replace("+", "plus").replace("-", "minus")
        rows.append({
            "sample": sample,
            "window": window,
            "k_24fps": k,
            "expected_delta_25fps": -k * 25.0 / 24.0,
            "clean_global": {"offset_25fps": clean_json["av_offset_frames"], "confidence": clean_json["confidence"]},
            "edited_global": {"offset_25fps": edit_json["av_offset_frames"], "confidence": edit_json["confidence"]},
            "audio_period": audio_period_25fps(sample, start, end),
            "segments": segments,
            "conclusion": d6_conclusion(in_window),
            "plot": save_d6_plot(label, segments),
        })
    return {"syncnet_reconstruction": "all D6 clean/edited global and 0.5-second segment offsets/confidences matched existing gen_lipsync.json before diagnosis", "rows": rows}


def d087():
    distributions = []
    for sample in SAMPLES:
        values = np.asarray(lipsync_json(lipsync_dir(sample))["seg_conf"], dtype=float)
        values = values[np.isfinite(values)]
        distributions.append({
            "sample": sample,
            "median": float(np.median(values)),
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "count": int(len(values)),
        })
    sample = "CRB100_087"
    frames = np.load(OUT / "clean" / sample / "clean.npy", mmap_mode="r")
    keypoints = np.load(OUT / "clean" / sample / "evaluators/dwpose.npz")
    screenshots = []
    for index in (96, 120, 192):
        image = cv2.cvtColor(np.asarray(frames[index]), cv2.COLOR_RGB2BGR)
        mouth = keypoints["face"][index, 60:68]
        if np.isfinite(mouth).all():
            x0, y0 = np.floor(mouth.min(axis=0) * np.asarray([image.shape[1], image.shape[0]])).astype(int)
            x1, y1 = np.ceil(mouth.max(axis=0) * np.asarray([image.shape[1], image.shape[0]])).astype(int)
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
        destination = MEDIA / f"d087_clean_frame_{index:03d}.png"
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_PNG_COMPRESSION, 8])
        scores = keypoints["face_score"][index, [61, 67, 62, 66, 63, 65, 36, 45]]
        screenshots.append({
            "source_frame_24fps": index,
            "required_landmark_score_min": float(scores.min()),
            "required_landmarks_all_at_least_0_3": bool(np.all(scores >= .3)),
            "path": str(destination.relative_to(ROOT)),
        })
    return {"all_samples_confidence_distribution": distributions, "clean_frame_screenshots": screenshots}


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    if DIAG_JSON.exists() or MEDIA.exists():
        raise FileExistsError("new D1′/D2′/D6 outputs already exist; refusing overwrite")
    d1 = d1_corrected()
    MEDIA.mkdir(parents=True)
    try:
        d2 = d2_corrected()
        d6_result = d6()
        d087_result = d087()
        result = {"D1_prime_posthoc": d1, "D2_prime_posthoc": d2, "D6_posthoc": d6_result, "CRB100_087_posthoc": d087_result}
        DIAG_JSON.write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    except Exception:
        for path in MEDIA.iterdir():
            path.unlink()
        MEDIA.rmdir()
        raise
    print(DIAG_JSON)


if __name__ == "__main__":
    main()
