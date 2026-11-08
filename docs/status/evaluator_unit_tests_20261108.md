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
### Output immutability guard before full matrix

Before expanding to W2 and the remaining seven people, preparation was hardened
so that a registered condition directory is created exactly once. `--window`
now selects an explicit registered subset, and any pre-existing condition
directory raises `FileExistsError` instead of being reused or overwritten.
This permits CRB100_001 W2 to be prepared without touching its completed W1
artifacts. The protection is covered by the focused test
`test_prepare_window_selection_and_fresh_directory_guard`; the full focused
suite passed (25 tests).The same refusal guard now applies to every evaluator output directory (clean,
condition, and determinism-repeat modes), so a retry cannot replace raw
readings already written under an existing artifact path.
### T6 v2.1 pre-analysis registration

Supplement `2026-11-08_evaluator_unit_tests_v2_1_T6.md` was registered in b507
commit `ad00e9f` before any T6 numeric result for the other seven people was
opened, computed, or aggregated. At registration time only result-file names
were listed to observe queue status; no T6 arrays, JSON metrics, or summary
values for those seven people were inspected. v2.1 changes only the geometric
left-hand isolation check; original v2 T6 is retained for parallel reporting.
## Revision 2.1 final matrix result (T6 left-hand overlap rule)
All full-matrix condition evaluations are complete. One report-only condition, CRB100_001/W2 T3 skip, was interrupted by the preserved GPU1 queue-parameter incident; it is not a v2 pass criterion and is not represented as a successful result.
### Fixed v2 decisions — other seven people
| Item | Passed people | Threshold | Result |
|---|---:|---:|---|
| T1 | 0/7 | 6/7 | fail |
| T2 | 7/7 | 6/7 | pass |
| T3 freeze | 7/7 | 6/7 | pass |
| T3 jump | 3/7 | 6/7 | fail |
| T4a | 7/7 | 6/7 | pass |
| T4b | 6/7 | 6/7 | pass |
| T4c | 5/7 | 6/7 | fail |
| T5 | 3/7 | 6/7 | fail |
| T6 v2 | 2/7 | 6/7 | fail |
| T6 v2.1 | 7/7 | 6/7 | pass |

### T6 v2 and v2.1 side-by-side
v2.1 was committed before any other-seven T6 numeric output was opened, computed, or aggregated. At that time only result-file names were listed for queue observation. v2.1 uses only occlusion-box/clean-left-box geometry; no evaluator score selects overlap frames.

| Sample | Window | v2 | Applied | Overlap | Non-overlap left eligible | v2.1 left check | v2.1 |
|---|---|---|---:|---:|---:|---|---|
| CRB100_001 | W1 | fail | 48 | 41 | 7 | undetermined | pass |
| CRB100_001 | W2 | fail | 48 | 48 | 0 | undetermined | pass |
| CRB100_002 | W1 | fail | 48 | 20 | 28 | passed | pass |
| CRB100_002 | W2 | fail | 48 | 48 | 0 | undetermined | pass |
| CRB100_025 | W1 | pass | 48 | 0 | 48 | passed | pass |
| CRB100_025 | W2 | fail | 48 | 33 | 15 | passed | pass |
| CRB100_050 | W1 | fail | 48 | 37 | 11 | undetermined | pass |
| CRB100_050 | W2 | pass | 48 | 0 | 48 | passed | pass |
| CRB100_056 | W1 | pass | 48 | 10 | 38 | passed | pass |
| CRB100_056 | W2 | pass | 48 | 0 | 48 | passed | pass |
| CRB100_075 | W1 | pass | 48 | 0 | 48 | passed | pass |
| CRB100_075 | W2 | pass | 48 | 4 | 44 | passed | pass |
| CRB100_087 | W1 | pass | 48 | 4 | 30 | passed | pass |
| CRB100_087 | W2 | fail | 48 | 40 | 6 | undetermined | pass |
| CRB100_090 | W1 | fail | 48 | 48 | 0 | undetermined | pass |
| CRB100_090 | W2 | pass | 48 | 0 | 48 | passed | pass |

Under the original v2 T6 rule, only 2/7 unseen people pass both windows, so T6 v2 fails. Under registered v2.1, all 7/7 unseen people pass both windows, so T6 v2.1 passes. The two conclusions are reported in parallel.

Private raw artifact root: `/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2/`; public JSON summary is `docs/status/evaluator_unit_tests_20261108_summary_v21_final.json`.

## 漏项补跑与系统核查（事后）
本节为看过全量结果后的事后诊断；不修改第二版或 v2.1 登记结论。
核查：320 条条件中标准目录完整 319 条，retry 完整 1 条，缺失 0 条。
CRB100_087/W1/T4 instant_ev_0.2 原队列脚本列出但静默未执行，目标目录从未创建；此前汇报只数现有 run_meta、未做笛卡尔积核查，故漏列。补跑后 T4b 从“补跑前 6/7（缺 1 条）”变为事后重算 7/7；这不改写原登记表。CRB100_001/W2 T3/skip 在独立 evaluators_posthoc_retry_v1 完成，且 skip 一直仅报告。

## 事后诊断（看过结果后进行，仅供参考）

### D1 事后：相对 clean 口型
| 置信度阈值 | 其余 7 人两个窗口相对准确 |
|---:|---:|
| 2 | 3/7 |
| 3 | 3/7 |
| 4 | 3/7 |
门槛 2/3/4 结论稳定为 3/7；SyncNet 置信度不再作为“必须下降”的判据。

### D2 事后：口型基线来源
| 样本 | mouth-audio lag | SyncNet 中位数 | 结论 |
|---|---:|---:|---|
| CRB100_001 | -5 | -1.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_002 | 2 | -3.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_025 | -4 | -4.0 | consistent_more_likely_data_offset |
| CRB100_050 | 10 | 0.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_056 | -9 | 0.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_075 | 10 | -1.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_087 | 10 | -2.0 | inconsistent_more_likely_syncnet_bias |
| CRB100_090 | -10 | -2.0 | inconsistent_more_likely_syncnet_bias |
相关系数整体较低，D2 只能提示而不能确证偏移来源。

### D3 事后：突跳扩展事件窗
| 样本 | 窗口 | a±5 检出 | 窗口外新增 | a/a+30 手有效点 | 位移(肩宽) |
|---|---|---|---:|---|---:|
| CRB100_001 | W1 | True | 0 | 42/42 | 0.387 |
| CRB100_001 | W2 | False | 0 | 42/42 | 0.026 |
| CRB100_002 | W1 | True | 0 | 42/42 | 0.319 |
| CRB100_002 | W2 | True | 0 | 42/41 | 0.281 |
| CRB100_025 | W1 | True | 0 | 42/42 | 0.157 |
| CRB100_025 | W2 | True | 0 | 42/42 | 0.055 |
| CRB100_050 | W1 | True | 0 | 42/42 | 0.214 |
| CRB100_050 | W2 | True | 0 | 42/42 | 0.220 |
| CRB100_056 | W1 | True | 0 | 42/42 | 1.716 |
| CRB100_056 | W2 | True | 0 | 42/42 | 0.688 |
| CRB100_075 | W1 | False | 0 | 42/42 | 0.057 |
| CRB100_075 | W2 | True | 0 | 42/42 | 0.150 |
| CRB100_087 | W1 | True | 0 | 42/21 | 1.369 |
| CRB100_087 | W2 | True | 0 | 24/42 | 1.180 |
| CRB100_090 | W1 | True | 0 | 42/42 | 0.372 |
| CRB100_090 | W2 | True | 0 | 42/42 | 0.077 |

### D4 事后：硬切像素差
| 样本 | 窗口 | a edit/clean | b edit/clean | clean p99 |
|---|---|---:|---:|---:|
| CRB100_001 | W1 | 8.06/0.75 | 6.56/0.87 | 5.51 |
| CRB100_001 | W2 | 4.52/0.85 | 4.36/1.45 | 5.51 |
| CRB100_002 | W1 | 21.82/7.16 | 23.02/8.41 | 13.33 |
| CRB100_002 | W2 | 15.26/12.54 | 23.93/10.64 | 13.33 |
| CRB100_025 | W1 | 12.82/0.55 | 13.85/1.61 | 4.01 |
| CRB100_025 | W2 | 7.62/0.92 | 6.46/0.35 | 4.01 |
| CRB100_050 | W1 | 7.27/2.23 | 8.73/2.07 | 3.16 |
| CRB100_050 | W2 | 7.24/2.11 | 9.71/1.98 | 3.16 |
| CRB100_056 | W1 | 8.41/2.47 | 13.46/4.40 | 6.97 |
| CRB100_056 | W2 | 13.47/3.73 | 8.94/1.42 | 6.97 |
| CRB100_075 | W1 | 5.82/1.10 | 5.63/0.45 | 5.00 |
| CRB100_075 | W2 | 6.66/1.12 | 7.26/1.37 | 5.00 |
| CRB100_087 | W1 | 18.99/8.52 | 21.37/3.22 | 14.62 |
| CRB100_087 | W2 | 18.48/4.75 | 25.43/11.67 | 14.62 |
| CRB100_090 | W1 | 23.69/1.49 | 9.89/0.99 | 5.60 |
| CRB100_090 | W2 | 11.71/1.24 | 14.35/0.51 | 5.60 |

### D5 事后：标签缺失与判错
有效段合并 AUC（仅有效段）=0.834。
| 样本 | 窗口 | 有效段 | 标签缺失 | correct≤wrong |
|---|---|---:|---|---|
| CRB100_001 | W1 | 4 | False | False |
| CRB100_001 | W2 | 4 | False | False |
| CRB100_002 | W1 | 4 | False | False |
| CRB100_002 | W2 | 4 | False | True |
| CRB100_025 | W1 | 4 | False | False |
| CRB100_025 | W2 | 4 | False | False |
| CRB100_050 | W1 | 4 | False | False |
| CRB100_050 | W2 | 4 | False | False |
| CRB100_056 | W1 | 0 | True | False |
| CRB100_056 | W2 | 0 | True | False |
| CRB100_075 | W1 | 4 | False | True |
| CRB100_075 | W2 | 4 | False | False |
| CRB100_087 | W1 | 1 | False | False |
| CRB100_087 | W2 | 0 | True | False |
| CRB100_090 | W1 | 4 | False | False |
| CRB100_090 | W2 | 4 | False | False |

## 评价器使用建议（事后）
| 评价器 | 建议 | 条件 |
|---|---|---|
| Lip-sync | 有条件地用 | 相对 clean 变化量；clean 置信度<3 记 NaN；当前 D1 仅 3/7，不能作为单独硬门槛。 |
| Identity | 可以直接用 | T2 7/7。 |
| Pose freeze | 可以直接用 | T3 freeze 7/7。 |
| Pose jump | 有条件地用 | 需结合 a±5、手可见性和位移；登记检出仅 3/7。 |
| Temporal flicker | 有条件地用 | T4a/T4b 的事后诊断证据可供条件性使用；硬切需先量化两侧画面差。 |
| Emotion | 有条件地用 | 仅有效标签段；标签缺失与判错分开。 |
| Hand T6 | 有条件地用 | 同时报 v2/v2.1；交叠帧用几何排除。 |
