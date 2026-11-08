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
OUT = Path('/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108')


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


def right_hand_visibility(path, n=240):
    """Per-frame mean visibility of only evaluator regions labeled right-hand."""
    z = np.load(path / 'hand_joint_visibility.npz')
    out = np.full(n, np.nan, dtype=float)
    f = z['region_frame'].astype(int)
    is_right = z['region_is_right'].astype(bool)
    scores = z['region_joint_visibility'].astype(float).mean(axis=1)
    for frame in np.unique(f[is_right]):
        out[frame] = finite_mean(scores[(f == frame) & is_right])
    return out


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
                detail = {
                    'k_24fps': k,
                    'expected_offset_25fps': expected,
                    'window_median_offset': finite_median(offsets[inside]),
                    'exterior_good_fraction': ratio(np.abs(offsets[outside]) <= 1),
                    'window_confidence_clean': finite_mean(clean_conf[inside]),
                    'window_confidence_edited': finite_mean(conf[inside]),
                }
                detail['required'] = abs(k) in (4, 8)
                detail['passed'] = (not detail['required']) or (
                    np.isfinite(detail['window_median_offset']) and
                    abs(detail['window_median_offset'] - expected) <= 1 and
                    detail['exterior_good_fraction'] >= .90 and
                    detail['window_confidence_edited'] < detail['window_confidence_clean']
                )
                per_k.append(detail)
            rows.append({'sample': sample, 'window': window, 'levels': per_k,
                         'passed': all(r['passed'] for r in per_k if r['required'])})
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


def t3(samples, windows):
    rows=[]
    for sample in samples:
        clean=pose(base(sample))
        for window,(a,b) in windows.items():
            for level in ('freeze','skip','jump'):
                p=cond('T3',sample,window,level)
                if not has_run(p): continue
                z=pose(p); all_flags=[k for k in z.files if k.startswith('gen_flag_')]
                if level=='freeze':
                    cover=ratio(np.asarray(z['gen_flag_frozen'],float)[a:a+12])
                    outside=np.ones(240,dtype=bool); outside[max(0,a-3):min(240,a+15)]=False
                    new=sum(count_flags(z,k,outside)-count_flags(clean,k,outside) for k in ['gen_flag_frozen'])
                    passed=cover>=.80 and new<=0
                    metric={'frozen_coverage':cover,'new_frozen_outside_band':new}
                else:
                    lo,hi=max(0,a-2),min(240,a+3)
                    found=bool(np.asarray(z['gen_flag_motion_jump'],bool)[lo:hi].any())
                    total_new=sum(max(0,int(np.asarray(z[k],bool).sum())-int(np.asarray(clean[k],bool).sum())) for k in all_flags)
                    passed=found and total_new<=1
                    metric={'motion_jump_near_a':found,'new_flags_total':total_new}
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
                z=temporal(p); flick.append({'e':e,'mean_flicker_score':finite_mean(z['flicker_score'][inside]),'window_flag_fraction':ratio(z['flicker_flag'][inside]),'exterior_flag_fraction':ratio(z['flicker_flag'][outside])})
            monotonic=len(flick)==3 and all(flick[i]['mean_flicker_score']<flick[i+1]['mean_flicker_score'] for i in range(2))
            point1=next((x for x in flick if x['e']==.1),None)
            flick_pass=bool(point1 and monotonic and point1['window_flag_fraction']>=.5 and point1['exterior_flag_fraction']<=ratio(clean['flicker_flag'][outside])+.02)
            p=cond('T4',sample,window,'hard_cut'); hard=None
            if has_run(p):
                z=temporal(p); frames=np.flatnonzero(z['boundary_flag'])+1
                hard={'boundary_frames':frames.tolist(),'passed':bool(np.any(np.abs(frames-a)<=1))}
            rows.append({'sample':sample,'window':window,'flicker_levels':flick,'flicker_passed':flick_pass,'hard_cut':hard,'passed':flick_pass and bool(hard and hard['passed'])})
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
        clean=right_hand_visibility(base(sample))
        for window,(a,b) in windows.items():
            p=cond('T6',sample,window,'right_hand_gray')
            if not has_run(p): continue
            edited=right_hand_visibility(p); inside=frame_mask(240,a,b); outside=frame_mask(240,a,b,exterior=True)
            drop=finite_mean(edited[inside])-finite_mean(clean[inside]); exterior=finite_mean(edited[outside])-finite_mean(clean[outside])
            rows.append({'sample':sample,'window':window,'right_hand_window_delta':drop,'right_hand_exterior_delta':exterior,'passed':np.isfinite(drop) and drop<=-.30 and abs(exterior)<=.05})
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
