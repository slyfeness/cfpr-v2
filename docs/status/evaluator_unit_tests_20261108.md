# Evaluator unit tests — pilot stop record (2026-11-08)

Status: **in progress; stopped after the required CRB100_001 × W1 pilot.**

The task requires A--C and preview inspection to pass before the other seven
people or W2 may start.  That gate was not met.  No thresholds, samples,
windows, evaluator references, or evaluator logic were changed to alter this
outcome.  No generator or training job was run.

## Exact pilot results

| Test | Result | Evidence |
|---|---|---|
| T1 Lip-sync | failed | All required offsets had the expected direction and were within one 25-fps frame, but exterior `|offset| <= 1` was 0.8571, below the pre-registered 0.90. |
| T2 Identity | passed for this pilot only | ArcFace window delta `-0.779776`; exterior delta `0.0`; DINO torso/hair window deltas `-0.3207/-0.463975`. |
| T3 Pose | failed | Freeze passed (coverage 1.0, no new frozen flag outside band); skip had no `motion_jump` in `a±2`; jump had a nearby flag but five new flags, above the allowed one. |
| T4 Temporal | failed | Mean flicker scores for 0.05/0.1/0.2 EV were 0.11062/0.13404/0.11148, not strictly monotonic; 0.1 EV flag fraction was 0.0.  Hard-cut boundary frames were 96 and 144, so the hard-cut subtest passed. |
| T5 Emotion | passed for this pilot only | Window means: correct 0.899689 > wrong 0.252315; pooled window AUC 1.0.  Targets are Meta automatic labels, not human ground truth. |
| T6 Hand | failed | Right-hand visibility window delta was `+0.130089`, not the required <= -0.30; exterior delta 0.0. |

T1 audio intervention validation: each W1 shift has an immutable
`audio_validation.json` sidecar.  The `k=+4` sidecar recovered `+2667` samples,
the registered rounded `4*16000/24`, and verified byte-identical exterior audio.

## Determinism (C)

Clean was run twice into separate `clean_repeats/CRB100_001/repeat{1,2}`
directories.  All seven NPZ evaluator outputs were elementwise equal with NaNs
treated as equal.  Two JSON files differed only in timing metadata:
`run_meta.json` wall-clock fields and temporal evidence's `seconds`; no score
array differed.

## Stop decision

The failed items are evaluator/test response failures (category A unless later
pixel evidence contradicts it), not grounds to retune thresholds.  Per the
task's D gate, the remaining seven people, W2, preview media, and full decision
table were not started.

## RAVDESS

`scripts/evaluators/ravdess_face_check.py` and historical evidence were found,
but no RAVDESS raw-video root was found under `/data/lanyi`; it was not rerun and
no new data was downloaded.  The historical status document records AUC 0.9253.

## Artifact root

`/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108/`

This root contains raw outputs, edit metadata, pilot log, and
`clean_determinism_check.json`; none are included in the repository.
