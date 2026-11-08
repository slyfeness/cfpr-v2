import json

import numpy as np

from scripts.evaluator_unit_tests import analyze


def _run(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "run_meta.json").write_text("{}")


def _lipsync(path, offsets, conf):
    (path / "lipsync").mkdir(parents=True, exist_ok=True)
    (path / "lipsync" / "gen_lipsync.json").write_text(json.dumps({"seg_offsets": offsets, "seg_conf": conf}))


def _temporal(path, *, level, score, flags=None, boundary=None):
    (path / "temporal").mkdir(parents=True, exist_ok=True)
    np.savez(path / "temporal" / "gen_temporal_context.npz", flicker_level=np.asarray(level, float), flicker_score=np.asarray(score, float), flicker_flag=np.asarray(flags if flags is not None else np.zeros(240, bool)), boundary_flag=np.asarray(boundary if boundary is not None else np.zeros(239, bool)))


def test_t1_compares_window_exterior_to_the_same_clean_segments(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "OUT", tmp_path)
    clean = tmp_path / "clean" / "s" / "evaluators"
    clean_offsets = [-6, -6, -6] + [0] * 17
    _lipsync(clean, clean_offsets, [0.8] * 20)
    edit = tmp_path / "T1" / "s" / "W1" / "k_+4" / "evaluators"
    _run(edit); _lipsync(edit, clean_offsets[:8] + [-4] * 4 + clean_offsets[12:], [0.3] * 20)
    row = analyze.t1(["s"], {"W1": (96, 144)})[0]
    assert row["levels"][0]["exterior_same_as_clean_fraction"] == 1.0
    assert row["passed"] is True


def test_t3_jump_accepts_jerk_and_excludes_the_registered_event_band(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "OUT", tmp_path)
    clean = tmp_path / "clean" / "s" / "evaluators" / "pose"; clean.mkdir(parents=True)
    z = dict(gen_flag_motion_jump=np.zeros(240, bool), gen_flag_jerk=np.zeros(240, bool), gen_flag_frozen=np.zeros(240, bool))
    np.savez(clean / "pose.npz", **z)
    edit = tmp_path / "T3" / "s" / "W1" / "jump" / "evaluators" / "pose"; edit.mkdir(parents=True)
    jumped = dict(z); jumped["gen_flag_jerk"] = z["gen_flag_jerk"].copy(); jumped["gen_flag_jerk"][96] = True
    np.savez(edit / "pose.npz", **jumped); (edit.parent / "run_meta.json").write_text("{}")
    row = analyze.t3(["s"], {"W1": (96, 144)})[0]
    assert row["motion_jump_or_jerk_near_a"] is True
    assert row["new_flags_outside_event_band"] == 0
    assert row["passed"] is True


def test_t4a_uses_continuous_flicker_level_not_spike_score(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "OUT", tmp_path)
    clean = tmp_path / "clean" / "s" / "evaluators"; _temporal(clean, level=np.zeros(240), score=np.zeros(240))
    for ev, level in ((.05, .01), (.1, .02), (.2, .04)):
        edit = tmp_path / "T4" / "s" / "W1" / f"ev_{ev:g}" / "evaluators"
        _run(edit); _temporal(edit, level=np.r_[np.zeros(96), np.full(48, level), np.zeros(96)], score=np.r_[np.zeros(96), np.full(48, .5-ev), np.zeros(96)])
    hard = tmp_path / "T4" / "s" / "W1" / "hard_cut" / "evaluators"
    _run(hard); boundary=np.zeros(239,bool); boundary[[95,143]]=True; _temporal(hard, level=np.zeros(240), score=np.zeros(240), boundary=boundary)
    row = analyze.t4(["s"], {"W1": (96, 144)})[0]
    assert row["t4a_passed"] is True
    assert row["hard_cut"]["passed"] is True


def test_t6_uses_expected_right_hand_visibility_and_missingness(tmp_path, monkeypatch):
    monkeypatch.setattr(analyze, "OUT", tmp_path)
    def hand(path, right, left, missing):
        path.mkdir(parents=True, exist_ok=True)
        np.savez(path / "hand_joint_visibility.npz", vis_right=np.asarray(right, float), vis_left=np.asarray(left, float), missing_right=np.asarray(missing, float), missing_left=np.zeros(240, float), visible_joint_fraction=np.asarray(left, float))
    hand(tmp_path / "clean" / "s" / "evaluators", np.full(240, .8), np.full(240, .6), np.zeros(240))
    right=np.full(240,.8);right[96:144]=.1;left=np.full(240,.6);left[96:144]=.55;missing=np.zeros(240);missing[96:144]=1
    edit=tmp_path / "T6" / "s" / "W1" / "right_hand_gray" / "evaluators"; hand(edit,right,left,missing);(edit / "run_meta.json").write_text("{}")
    row=analyze.t6(["s"], {"W1": (96,144)})[0]
    assert row["vis_right_window_delta"] <= -.3
    assert row["missing_right_window_fraction"] == 1.0
    assert row["passed"] is True
