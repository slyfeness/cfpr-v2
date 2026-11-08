#!/usr/bin/env python3
"""Fixed, NaN-aware analysis for the 2026-11-08 evaluator unit-test protocol.

This module contains the pre-registered decision rules only.  It deliberately
does not select samples, windows, strengths, thresholds, or reference files.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path('/data/lanyi/projects/cfpr-v2')
OUT = Path('/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2')


def load_json(path):
    with path.open() as f:
        return json.load(f)


def finite_mean(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def finite_median(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else float('nan')


def ratio(values):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float('nan')


def frame_mask(n, start, end, *, exterior=False, transition=6):
    i = np.arange(n)
    if exterior:
        return (i < start - transition) | (i >= end + transition)
    return (i >= start) & (i < end)


def seg_mask(n, start, end, *, exterior=False, transition=6):
    # Every evaluator segment is 0.5 seconds = 12 source frames.  A segment is
    # usable as exterior only when it does not intersect the transition band.
    left = np.arange(n) * 12
    right = left + 12
    if exterior:
        return (right <= start - transition) | (left >= end + transition)
    return (left < end) & (right > start)


def base(sample):
    return OUT / 'clean' / sample / 'evaluators'


def cond(test, sample, window, level):
    return OUT / test / sample / window / level / 'evaluators'


def has_run(path):
    return (path / 'run_meta.json').exists()


def lipsync(path):
    return load_json(path / 'lipsync' / 'gen_lipsync.json')


def arcface(path):
    z = np.load(path / 'face_gen.npz')
    return z['frame'].astype(int), z['arcface'].astype(float)


def temporal(path):
    return np.load(path / 'temporal' / 'gen_temporal_context.npz')


def pose(path):
    return np.load(path / 'pose' / 'pose.npz')


def emotion(path):
    return np.asarray(load_json(path / 'emotion.json')['videos']['gen']['consistency_seg'], dtype=float)


def expected_hand_visibility(path, n=240):
    z = np.load(path / 'hand_joint_visibility.npz')
    required = ('vis_left', 'vis_right', 'missing_left', 'missing_right')
    if any(key not in z.files for key in required):
        raise ValueError('v2 expected-hand fields are required; legacy aggregate regions cannot score T6')
    values = {key: np.asarray(z[key], dtype=float) for key in required}
    if any(value.shape != (n,) for value in values.values()):
        raise ValueError('expected-hand arrays must be per-frame length n')
    legacy = np.asarray(z['visible_joint_fraction'], dtype=float) if 'visible_joint_fraction' in z.files else np.full(n, np.nan)
    return values, legacy


def t1(samples, windows):
    rows = []
    for sample in samples:
        clean = lipsync(base(sample))
        clean_offsets = np.asarray(clean['seg_offsets'], dtype=float)
        clean_conf = np.asarray(clean['seg_conf'], dtype=float)
        for window, (a, b) in windows.items():
            inside, outside = seg_mask(len(clean_offsets), a, b), seg_mask(len(clean_offsets), a, b, exterior=True)
            per_k = []
            for k in (-8, -4, -2, 2, 4, 8):
                p = cond('T1', sample, window, f'k_{k:+d}')
                if not has_run(p):
                    continue
                got = lipsync(p)
                offsets = np.asarray(got['seg_offsets'], dtype=float)
                conf = np.asarray(got['seg_conf'], dtype=float)
                expected = -k * 25.0 / 24.0
                same = np.abs(offsets[outside] - clean_offsets[outside]) <= 1
                detail = {'k_24fps': k, 'expected_offset_25fps': expected,
                    'window_median_offset': finite_median(offsets[inside]),
                    'exterior_same_as_clean_fraction': ratio(same),
                    'window_confidence_clean': finite_mean(clean_conf[inside]),
                    'window_confidence_edited': finite_mean(conf[inside]), 'required': abs(k) in (4, 8)}
                detail['passed'] = (not detail['required']) or (np.isfinite(detail['window_median_offset']) and
                    abs(detail['window_median_offset'] - expected) <= 1 and detail['exterior_same_as_clean_fraction'] >= .90 and
                    detail['window_confidence_edited'] < detail['window_confidence_clean'])
                per_k.append(detail)
            rows.append({'sample': sample, 'window': window, 'levels': per_k,
                         'passed': bool(per_k) and all(r['passed'] for r in per_k if r['required'])})
    return rows

def t2(samples, windows):
    rows = []
    for sample in samples:
        cb = base(sample); frames, clean_arc = arcface(cb)
        dclean = load_json(cb / 'dino.json')['gen']
        for window, (a, b) in windows.items():
            p = cond('T2', sample, window, 'swap_next')
            if not has_run(p): continue
            _, edit_arc = arcface(p); dedit = load_json(p / 'dino.json')['gen']
            inside = (frames >= a) & (frames < b)
            outside = (frames < a - 6) | (frames >= b + 6)
            seg_in, seg_out = seg_mask(20,a,b),seg_mask(20,a,b,exterior=True)
            drop = finite_mean(edit_arc[inside]) - finite_mean(clean_arc[inside])
            exterior = finite_mean(edit_arc[outside] - clean_arc[outside])
            rows.append({'sample':sample,'window':window,'arcface_window_delta':drop,
                         'arcface_exterior_delta':exterior,
                         'dino_torso_window_delta':finite_mean(np.asarray(dedit['seg_torso_nn'])[seg_in])-finite_mean(np.asarray(dclean['seg_torso_nn'])[seg_in]),
                         'dino_hair_window_delta':finite_mean(np.asarray(dedit['seg_hair_nn'])[seg_in])-finite_mean(np.asarray(dclean['seg_hair_nn'])[seg_in]),
                         'dino_torso_exterior_delta':finite_mean(np.asarray(dedit['seg_torso_nn'])[seg_out])-finite_mean(np.asarray(dclean['seg_torso_nn'])[seg_out]),
                         'passed':np.isfinite(drop) and drop <= -.30 and abs(exterior) <= .02})
    return rows


def count_flags(z, name, mask):
    return int(np.asarray(z[name], dtype=bool)[mask].sum())


def _new_flags_outside(clean, edited, keys, outside):
    return sum(int((np.asarray(edited[k], bool) & ~np.asarray(clean[k], bool) & outside).sum()) for k in keys)


def t3(samples, windows):
    rows=[]
    for sample in samples:
        clean=pose(base(sample))
        for window,(a,b) in windows.items():
            for level in ('freeze','skip','jump'):
                p=cond('T3',sample,window,level)
                if not has_run(p): continue
                z=pose(p); all_flags=[k for k in z.files if k.startswith('gen_flag_') and k in clean.files]
                if level=='freeze':
                    cover=ratio(np.asarray(z['gen_flag_frozen'],float)[a:a+12])
                    outside=np.ones(240,dtype=bool); outside[max(0,a-3):min(240,a+15)]=False
                    new=_new_flags_outside(clean,z,['gen_flag_frozen'],outside)
                    metric={'frozen_coverage':cover,'new_frozen_outside_band':new}; passed=cover>=.80 and new<=0
                elif level=='jump':
                    event=np.zeros(240,bool); event[max(0,a-2):min(240,a+3)]=True
                    found=bool(np.asarray(z['gen_flag_motion_jump'],bool)[event].any() or np.asarray(z['gen_flag_jerk'],bool)[event].any())
                    new=_new_flags_outside(clean,z,all_flags,~event)
                    metric={'motion_jump_or_jerk_near_a':found,'new_flags_outside_event_band':new}; passed=found and new<=1
                else:
                    event_a=np.zeros(240,bool); event_b=np.zeros(240,bool); event_a[max(0,a-2):min(240,a+3)]=True; event_b[max(0,b-2):min(240,b+3)]=True
                    marker=np.asarray(z['gen_flag_motion_jump'],bool) | np.asarray(z['gen_flag_jerk'],bool)
                    metric={'motion_jump_or_jerk_near_a':bool(marker[event_a].any()),'motion_jump_or_jerk_near_b':bool(marker[event_b].any()),'reported_only':True}; passed=True
                rows.append({'sample':sample,'window':window,'level':level,**metric,'passed':passed})
    return rows

def t4(samples, windows):
    rows=[]
    for sample in samples:
        clean=temporal(base(sample))
        for window,(a,b) in windows.items():
            inside=frame_mask(240,a,b); outside=frame_mask(240,a,b,exterior=True)
            flick=[]
            for e in (.05,.1,.2):
                p=cond('T4',sample,window,f'ev_{e:g}')
                if not has_run(p): continue
                z=temporal(p)
                flick.append({'e':e,'mean_flicker_level_ev':finite_mean(z['flicker_level'][inside]),'mean_flicker_score':finite_mean(z['flicker_score'][inside]),'window_flag_fraction':ratio(z['flicker_flag'][inside]),'exterior_level_delta_ev':finite_mean(z['flicker_level'][outside])-finite_mean(clean['flicker_level'][outside])})
            monotonic=len(flick)==3 and all(flick[i]['mean_flicker_level_ev']<flick[i+1]['mean_flicker_level_ev'] for i in range(2))
            e05=next((x for x in flick if x['e']==.05),None)
            baseline_p99=float(np.nanquantile(np.asarray(clean['flicker_level'],float)[outside],.99))
            t4a=bool(e05 and monotonic and e05['mean_flicker_level_ev']>baseline_p99 and all(abs(x['exterior_level_delta_ev'])<=.005 for x in flick))
            instant=[]
            for e in (.1,.2,.4):
                p=cond('T4',sample,window,f'instant_ev_{e:g}')
                if not has_run(p): continue
                z=temporal(p); near=np.zeros(240,bool);near[max(0,a-1):min(240,a+2)]=True
                new_flags=int((np.asarray(z['flicker_flag'],bool) & ~np.asarray(clean['flicker_flag'],bool) & ~near).sum())
                instant.append({'e':e,'flicker_flag_near_a':bool(np.asarray(z['flicker_flag'],bool)[near].any()),'new_flags_outside_event_band':new_flags})
            needed=[x for x in instant if x['e'] in (.2,.4)]
            t4b=len(needed)==2 and all(x['flicker_flag_near_a'] and x['new_flags_outside_event_band']<=1 for x in needed)
            p=cond('T4',sample,window,'hard_cut'); hard=None
            if has_run(p):
                z=temporal(p)
                if len(z['boundary_flag']) != 239: raise AssertionError('boundary_flag must represent exactly 239 adjacent frame pairs')
                frames=np.flatnonzero(z['boundary_flag'])+1
                hard={'boundary_frames':frames.tolist(),'near_a':bool(np.any(np.abs(frames-a)<=1)),'near_b':bool(np.any(np.abs(frames-b)<=1))}
                hard['passed']=hard['near_a'] and hard['near_b']
            rows.append({'sample':sample,'window':window,'flicker_levels':flick,'instant_levels':instant,'clean_exterior_flicker_level_p99_ev':baseline_p99,'t4a_passed':t4a,'t4b_passed':t4b,'hard_cut':hard,'t4c_passed':bool(hard and hard['passed']),'passed':t4a and t4b and bool(hard and hard['passed'])})
    return rows

def auc_rank(pos, neg):
    x=np.concatenate([pos,neg]); labels=np.concatenate([np.ones(len(pos)),np.zeros(len(neg))]); good=np.isfinite(x); x,labels=x[good],labels[good]
    if not len(x) or not np.any(labels==1) or not np.any(labels==0): return float('nan')
    order=np.argsort(x); ranks=np.empty(len(x),float); ranks[order]=np.arange(1,len(x)+1)
    for value in np.unique(x):
        same=x==value; ranks[same]=ranks[same].mean()
    n1=int((labels==1).sum()); n0=int((labels==0).sum())
    return float((ranks[labels==1].sum()-n1*(n1+1)/2)/(n1*n0))


def t5(samples, windows):
    rows=[]; pooled_correct=[]; pooled_wrong=[]
    for sample in samples:
        for window,(a,b) in windows.items():
            c=cond('T5',sample,window,'correct'); w=cond('T5',sample,window,'wrong')
            if not (has_run(c) and has_run(w)): continue
            mask=seg_mask(20,a,b); correct,wrong=emotion(c)[mask],emotion(w)[mask]
            pooled_correct.extend(correct.tolist()); pooled_wrong.extend(wrong.tolist())
            cm,wm=finite_mean(correct),finite_mean(wrong)
            rows.append({'sample':sample,'window':window,'correct_window_mean':cm,'wrong_window_mean':wm,'passed':np.isfinite(cm) and np.isfinite(wm) and cm>wm})
    return rows, {'pooled_window_auc_correct_vs_wrong':auc_rank(np.asarray(pooled_correct),np.asarray(pooled_wrong))}


def t6(samples, windows):
    rows=[]
    for sample in samples:
        clean, clean_legacy = expected_hand_visibility(base(sample))
        for window,(a,b) in windows.items():
            p=cond('T6',sample,window,'right_hand_gray')
            if not has_run(p): continue
            edited, edited_legacy = expected_hand_visibility(p)
            inside=frame_mask(240,a,b); outside=frame_mask(240,a,b,exterior=True)
            vis_right_delta=finite_mean(edited['vis_right'][inside])-finite_mean(clean['vis_right'][inside])
            vis_left_delta=finite_mean(edited['vis_left'][inside])-finite_mean(clean['vis_left'][inside])
            vis_right_exterior_delta=finite_mean(edited['vis_right'][outside])-finite_mean(clean['vis_right'][outside])
            missing_right=ratio(edited['missing_right'][inside])
            old_fraction_delta=finite_mean(edited_legacy[inside])-finite_mean(clean_legacy[inside])
            passed=(np.isfinite(vis_right_delta) and vis_right_delta<=-.30) or (np.isfinite(missing_right) and missing_right>=.80)
            passed=bool(passed and np.isfinite(vis_left_delta) and abs(vis_left_delta)<=.10 and np.isfinite(vis_right_exterior_delta) and abs(vis_right_exterior_delta)<=.05)
            rows.append({'sample':sample,'window':window,'vis_right_window_delta':vis_right_delta,
                'missing_right_window_fraction':missing_right,'vis_left_window_delta':vis_left_delta,
                'vis_right_exterior_delta':vis_right_exterior_delta,'legacy_visible_joint_fraction_window_delta':old_fraction_delta,'passed':passed})
    return rows


def by_person_both_windows(rows, test):
    values={}
    for row in rows:
        if test=='T3': key=(row['sample'],row['window']); values.setdefault(key,[]).append(row['passed'])
        else: values[(row['sample'],row['window'])]=[row['passed']]
    person={}
    for sample in {k[0] for k in values}:
        person[sample]=all(all(values.get((sample,w),[False])) for w in ('W1','W2'))
    return {'per_person':person,'passed_people':sum(person.values()),'overall_passed':sum(person.values())>=7}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    cfg=load_json(ROOT/'scripts/evaluator_unit_tests/config.json'); samples=cfg['samples']; windows={k:tuple(v) for k,v in cfg['windows'].items()}
    result={'config':cfg,'analysis_conventions':{'nan_policy':'All aggregates exclude NaN; no NaN is replaced by zero.','frame_indexing':'0-based half-open windows; exterior excludes a six-frame band on either side.','segment_indexing':'12 frames per 0.5-second segment; a segment intersects a window when its half-open interval overlaps.'}}
    result['T1']=t1(samples,windows); result['T2']=t2(samples,windows); result['T3']=t3(samples,windows); result['T4']=t4(samples,windows); result['T5'],result['T5_pooled']=t5(samples,windows); result['T6']=t6(samples,windows)
    result['decision']={test:by_person_both_windows(result[test],test) for test in ('T1','T2','T3','T4','T5','T6')}
    path=Path(args.output)
    if path.exists(): raise FileExistsError(f'refusing to overwrite analysis output: {path}')
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=True))


if __name__=='__main__':
    main()
