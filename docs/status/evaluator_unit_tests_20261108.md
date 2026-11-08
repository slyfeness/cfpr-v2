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


## Revision 2 protocol — implementation stage (in progress)

The preceding sections are the immutable first-version CRB100_001 × W1 pilot record.
They are retained verbatim. Revision 2 uses the same eight samples and registered
windows, with the revised standards fixed in
`docs/tasks/2026-11-08_evaluator_unit_tests_v2.md` before any of the seven new
people are run. Its raw outputs are written to
`/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2/`, rather than
replacing the first-version artifact root.

### C1 hand-side verification before v2 execution

DWPose order is fixed as `hands[0] = person-left`, `hands[1] = person-right`.
Three independent pieces of evidence were recorded on CRB100_001 frame 0:

| Evidence | DWPose index 0 | DWPose index 1 |
|---|---:|---:|
| COCO-WholeBody left hand (91–111), mean aligned distance | 16.07 px | 55.32 px |
| COCO-WholeBody right hand (112–132), mean aligned distance | 60.76 px | 13.65 px |
| WiLoR false/left vs true/right best-box IoU | 0.858 / 0.334 | 0.465 / 0.792 |

The source COCO keypoint convention is left `91:112`, right `112:133`; the
raw source is named in `prepared_inputs_aligned_v1/CRB100_001/input_manifest.json`.
The manually inspectable annotated frame is
`evaluator_unit_tests_20261108_v2/preflight/CRB100_001_frame000_dwp_hands.png`.
Accordingly, all v2 T6 occlusions use index 1 and expected left/right boxes
are generated from clean DWPose only.

### Implemented v2 corrections and pre-run tests

- C2: windowed audio shifts now fill the vacated samples from adjacent real
  audio, never silence; outside-window samples remain byte-identical.
- C3/C4/C5/C6/C7: skip uses the fixed 12-frame mapping; jump accepts
  `motion_jump` or `jerk` and excludes its registered event band; T1 compares
  exterior offsets against the same clean segments; T4a uses `flicker_level`,
  T4b is a registered one-frame exposure test, and T4c requires both
  boundaries while asserting 239 pair records.
- T6 adds an expected-hand mode without changing legacy output fields:
  side-consistent IoU >= 0.3 association, expected-but-unmatched visibility
  0 and missingness 1, and NaN for hands not expected in a frame.
- Fresh focused test evidence before the pilot: `21 passed` from
  `tests/test_evaluator_unit_edits.py`,
  `tests/test_hand_visibility_expected.py`, and
  `tests/test_evaluator_unit_analysis_v2.py`.

The v2 CRB100_001 W1 queue is running at the time of this update. No
threshold, sample, window, reference file, evaluator weight, generator, or
training configuration has been changed.


### Revision 2 CRB100_001 × W1 pilot

All 20 registered conditions completed. Edit self-check: 20/20 passed. Clean
repeat5/repeat6 produced elementwise-equal (NaN-aware) arrays for DWPose,
ArcFace, SyncNet, pose, temporal evidence/context, and expected-hand WiLoR.
The actual pilot decisions are T1 pass, T2 pass, T3 freeze pass, T3 jump pass
(skip is report-only), T4a/T4b/T4c pass, T5 pass, and T6 fail. T6's expected
right-hand visibility fell by -0.8219 and missing-right fraction was 0.9792,
but left-hand visibility also fell by -0.4571 (required absolute change <=0.10).
This is retained as a control-isolation failure, not repaired by changing a
threshold, sample, or window. Preview assets are audit-only and listed in the
publication manifest; their pixels are never evaluator inputs.

### Revision 2 pilot audit, reproducibility, and publication preparation

This subsection supersedes the implementation-stage sentence that called the
queue “running”; the completed pilot result above is authoritative. The
pre-registered v2 decision logic was fixed before any unseen person was
started. No threshold, sample, window, model, or evaluator reference was
changed after observing this result.

- **T1:** every registered offset had relative exterior agreement 1.0 and the
  required direction/delta.
- **T2:** ArcFace target-window delta was -0.779776 and exterior delta was
  0.0.
- **T3:** freeze coverage was 1.0 with no exterior new frozen flag; the jump
  evidence lay near the registered event and was exterior-clean. The
  registered skip condition is reported but does not enter the v2 pass count.
- **T4:** flicker-level means were 0.032202, 0.047016, and 0.100055 for
  0.05/0.10/0.20 EV; the one-frame condition triggered; the hard-cut
  boundaries were 96 and 144 and all 239 adjacent-pair records were present.
- **T5:** correct/wrong means were 0.899689/0.252315; pooled AUC was 1.0.
- **T6:** right visibility delta was -0.821909 and right missing fraction
  0.979167, but left visibility delta was -0.457066. The required
  non-target isolation condition (absolute left change <= 0.10) therefore
  fails. This failure remains a result, rather than a basis to alter the
  intervention or threshold.

The clean determinism retries repeat1--repeat4 are retained in the private
artifact root as failed setup attempts: repeat1/2 targeted a nonexistent clean
emotion target; repeat3/4 omitted the loaded configuration in the repeat
path. They produced no accepted result. With those execution bugs fixed,
repeat5/repeat6 completed and `clean_determinism_check_v2.json` records
NaN-aware elementwise equality for every numeric evaluator output and exact
equality for identifiers.

The pilot generated 20 paired audit previews (40 files): an MP4 comparison and
a PNG contact sheet for each registered condition. They are derived only after
evaluation and are never evaluator inputs. The released copies were size-audited:
every MP4 is below 3 MB, every PNG is below 500 KB, and their total is
6,391,621 bytes. The assets contain the approved public preview frames only;
no arrays, audio, checkpoints, raw datasets, credentials, or full-resolution
source material are included.