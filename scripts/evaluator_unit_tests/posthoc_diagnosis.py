#!/usr/bin/env python3
"""Posthoc D1-D5 diagnosis; never changes registered decisions."""
import json, numpy as np
from pathlib import Path
from scipy.io import wavfile
from scripts.evaluator_unit_tests import analyze
ROOT=Path('/data/lanyi/projects/cfpr-v2'); OUT=Path('/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2'); cfg=json.loads((ROOT/'scripts/evaluator_unit_tests/config.json').read_text()); S=cfg['samples']; W=cfg['windows']
def lip(p): return json.load(open(p/'lipsync/gen_lipsync.json'))
def mask(a,b): return np.array([(i*12<b and (i+1)*12>a) for i in range(20)])
def d1():
 rows=[]; counts={}
 for th in (2,3,4):
  pp={}
  for s in S:
   ok=[]; und=False
   c=lip(OUT/'clean'/s/'evaluators')
   for w,(a,b) in W.items():
    seg=mask(a,b); co=np.array(c['seg_offsets'],float); cf=np.array(c['seg_conf'],float)
    wins=[]
    for k in (-8,-4,4,8):
     x=lip(OUT/'T1'/s/w/f'k_{k:+d}'/'evaluators'); delta=np.array(x['seg_offsets'],float)-co; valid=seg&(cf>=th)&np.isfinite(delta); n=int(valid.sum()); med=analyze.finite_median(delta[valid]); wins.append({'k':k,'valid_segments':n,'median_delta':med,'expected':-k*25/24,'abs_error':abs(med+k*25/24) if np.isfinite(med) else float('nan'),'relative_accurate':bool(n>=2 and abs(med+k*25/24)<=1)})
    good=all(x['relative_accurate'] for x in wins); und|=any(x['valid_segments']<2 for x in wins); ok.append(good); rows += [dict(sample=s,window=w,confidence_threshold=th,**x) for x in wins]
   pp[s]=bool(all(ok)) and not und
  counts[str(th)]={'other_seven_both_windows_accurate':sum(pp[x] for x in S[1:]),'per_person':pp}
 return {'rows':rows,'counts':counts}
def d2():
 rows=[]
 for s in S:
  z=np.load(OUT/'clean'/s/'evaluators/dwpose.npz'); f=z['face']; fs=z['face_score']; up=(f[:,62]+f[:,63])/2; lo=(f[:,66]+f[:,67])/2; eye=np.linalg.norm(f[:,36]-f[:,45],axis=1); mouth=np.linalg.norm(up-lo,axis=1)/np.maximum(eye,1e-6); mouth[~np.isfinite(mouth)]=np.nan
  sr,a=wavfile.read(ROOT.parent.parent/'outputs/cfpr-v2/prepared_inputs_aligned_v1'/s/'audio.wav'); a=a.astype(float); env=np.array([np.sqrt(np.mean(a[int(i*sr/24):int((i+1)*sr/24)]**2)) for i in range(240)])
  best=(float('-inf'),None)
  for lag in range(-10,11):
   x=mouth[max(0,lag):min(240,240+lag)]; y=env[max(0,-lag):min(240,240-lag)]; good=np.isfinite(x)&np.isfinite(y)
   if good.sum()>10:
    r=np.corrcoef(x[good],y[good])[0,1]
    if r>best[0]:best=(r,lag)
  so=np.nanmedian(np.asarray(lip(OUT/'clean'/s/'evaluators')['seg_offsets'],float)); rows.append({'sample':s,'mouth_audio_lag':best[1],'correlation':best[0],'syncnet_median_offset':so,'conclusion':'inconclusive' if best[1] is None else ('consistent_more_likely_data_offset' if abs(best[1]-so)<=1 else 'inconsistent_more_likely_syncnet_bias')})
 return rows
def d3():
 rows=[]
 for s in S:
  z=np.load(OUT/'clean'/s/'evaluators/dwpose.npz'); h,hs,b,bs=z['hands'],z['hands_score'],z['body'],z['body_score']
  for w,(a,e) in W.items():
   p=OUT/'T3'/s/w/'jump/evaluators/pose/pose.npz'
   if not p.exists():continue
   q=np.load(p); c=np.load(OUT/'clean'/s/'evaluators/pose/pose.npz'); flags=(q['gen_flag_motion_jump']|q['gen_flag_jerk']) & ~(c['gen_flag_motion_jump']|c['gen_flag_jerk']); outside=flags.copy();outside[max(0,a-5):a+6]=False
   va=[int((hs[t]>=.3).sum()) for t in (a,a+30)]; cen=[np.nanmean(h[t][hs[t]>=.3],axis=0) if va[i] else np.array([np.nan,np.nan]) for i,t in enumerate((a,a+30))]; sw=np.linalg.norm(b[a,2]-b[a,5]) if bs[a,2]>.3 and bs[a,5]>.3 else np.nan; disp=np.linalg.norm(cen[1]-cen[0])/sw if np.isfinite(sw) and sw>0 else np.nan
   rows.append({'sample':s,'window':w,'event_detected_a5':bool(flags[max(0,a-5):a+6].any()),'new_flags_outside_a5':int(outside.sum()),'hand_valid_a':va[0],'hand_valid_a30':va[1],'hand_centroid_displacement_shoulder':disp})
 return rows
def d4():
 rows=[]
 for s in S:
  clean=np.load(OUT/'clean'/s/'clean.npy',mmap_mode='r'); cd=np.abs(np.diff(clean.astype(float),axis=0)).mean((1,2,3)); p99=float(np.quantile(cd,.99))
  for w,(a,b) in W.items():
   x=np.load(OUT/'T4'/s/w/'hard_cut/frames.npy',mmap_mode='r'); ed=np.abs(np.diff(x.astype(float),axis=0)).mean((1,2,3)); rows.append({'sample':s,'window':w,'a_edit':float(ed[a-1]),'a_clean':float(cd[a-1]),'b_edit':float(ed[b-1]),'b_clean':float(cd[b-1]),'clean_p99':p99})
 return rows
def d5():
 rows=[]; pc=[];pw=[]
 for s in S:
  for w,(a,b) in W.items():
   seg=mask(a,b); c=np.asarray(json.load(open(OUT/'T5'/s/w/'correct/evaluators/emotion.json'))['videos']['gen']['consistency_seg'],float); q=np.asarray(json.load(open(OUT/'T5'/s/w/'wrong/evaluators/emotion.json'))['videos']['gen']['consistency_seg'],float); valid=np.isfinite(c)&np.isfinite(q)&seg;pc+=c[valid].tolist();pw+=q[valid].tolist();rows.append({'sample':s,'window':w,'valid_segments':int(valid.sum()),'correct_mean':analyze.finite_mean(c[valid]),'wrong_mean':analyze.finite_mean(q[valid]),'measured_failure':bool(valid.any() and analyze.finite_mean(c[valid])<=analyze.finite_mean(q[valid])),'label_missing':not bool(valid.any())})
 return {'rows':rows,'pooled_auc_valid_only':analyze.auc_rank(np.array(pc),np.array(pw))}
out={'D1_posthoc':d1(),'D2_posthoc':d2(),'D3_posthoc':d3(),'D4_posthoc':d4(),'D5_posthoc':d5()}
p=OUT/'posthoc_diagnosis_20261109.json'
if p.exists():raise FileExistsError(p)
p.write_text(json.dumps(out,indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x) if isinstance(x,np.floating) else int(x) if isinstance(x,np.integer) else str(x)))
print(p)