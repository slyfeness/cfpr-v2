#!/usr/bin/env python3
"""Prepare fixed evaluator unit-test counterfactuals (no model execution)."""
from __future__ import annotations
import argparse, io, json, os, subprocess, tarfile, time
from pathlib import Path
import cv2
import numpy as np
from scipy.io import wavfile
from .make_edits import (align_emotion_scores, apply_flicker_ev, assert_changed_frames,
    freeze_window, hard_cut_window, jump_frame, occlude_right_hand, replace_window,
    shift_audio_window, skip_forward_window, verify_audio_shift_xcorr, apply_single_frame_exposure_ev)

ROOT=Path('/data/lanyi/projects/cfpr-v2')
INPUT=Path('/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1')
OUT=Path('/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2')

def dump(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def decode_once(sample, target, cfg):
    if target.exists():
        x=np.load(target, mmap_mode='r')
        if x.shape==(cfg['frames'],768,768,3) and x.dtype==np.uint8:return target
        raise ValueError(f'Unexpected existing clean.npy {target}: {x.shape} {x.dtype}')
    cap=cv2.VideoCapture(str(INPUT/sample/'video.mp4')); rows=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        rows.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    x=np.stack(rows)
    if x.shape!=(cfg['frames'],768,768,3) or x.dtype!=np.uint8:raise ValueError(f'decode mismatch {sample}: {x.shape} {x.dtype}')
    target.parent.mkdir(parents=True,exist_ok=True); np.save(target,x); return target

def clean_audio(sample,cfg):
    sr,a=wavfile.read(INPUT/sample/'audio.wav')
    if sr!=cfg['audio_sample_rate'] or a.ndim!=1 or a.dtype!=np.int16:raise ValueError(f'audio contract failed {sample}: {sr} {a.shape} {a.dtype}')
    return a

def validate_existing_t1_audio(sample, window, level, cfg):
    """Append an immutable validation sidecar for a pre-existing T1 audio artifact."""
    if window not in cfg['windows'] or not level.startswith('k_'):
        raise ValueError('expected a registered T1 window and level such as W1 / k_+4')
    p=OUT/'T1'/sample/window/level; target=p/'audio_validation.json'
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing validation: {target}')
    sr,changed=wavfile.read(p/'audio.wav')
    clean=clean_audio(sample,cfg)
    if sr != cfg['audio_sample_rate'] or changed.dtype != np.int16 or changed.shape != clean.shape:
        raise ValueError('existing audio artifact violates the registered PCM contract')
    a,b=cfg['windows'][window]
    sample_a,sample_b=round(a*cfg['audio_sample_rate']/cfg['fps']),round(b*cfg['audio_sample_rate']/cfg['fps'])
    shift=round(int(level.split('_',1)[1])*cfg['audio_sample_rate']/cfg['fps'])
    dump({
        'artifact':str(p/'audio.wav'),
        'audio_exterior_unchanged':bool(np.array_equal(clean[:sample_a],changed[:sample_a]) and np.array_equal(clean[sample_b:],changed[sample_b:])),
        'audio_shift_xcorr':verify_audio_shift_xcorr(clean,changed,sample_a,sample_b,shift),
        'reason_for_sidecar':'artifact predates mandatory xcorr field in edit_meta.json; existing output is not overwritten',
    },target)

def expected_boxes_from_dwpose(hands, hand_scores, *, width, height, confidence=0.3, minimum_points=5, margin=0.2):
    """Return (left, right) xyxy boxes from DWPose's verified [left, right] order."""
    hands = np.asarray(hands, dtype=float)
    hand_scores = np.asarray(hand_scores, dtype=float)
    if hands.ndim != 4 or hands.shape[1:] != (2, 21, 2) or hand_scores.shape != hands.shape[:3]:
        raise ValueError("expected DWPose hands [T,2,21,2] and scores [T,2,21]")
    out = [np.full((len(hands), 4), np.nan, dtype=np.float32) for _ in range(2)]
    for frame in range(len(hands)):
        for side in (0, 1):
            points = hands[frame, side]
            valid = (hand_scores[frame, side] >= confidence) & np.isfinite(points).all(axis=1)
            if int(valid.sum()) < minimum_points:
                continue
            xy = points[valid].copy()
            if np.nanmax(np.abs(xy)) <= 1.5:
                xy[:, 0] *= width
                xy[:, 1] *= height
            x0, y0 = xy.min(axis=0); x1, y1 = xy.max(axis=0)
            dx, dy = max(1.0, x1 - x0), max(1.0, y1 - y0)
            out[side][frame] = [max(0., x0 - margin * dx), max(0., y0 - margin * dy),
                                min(float(width), x1 + margin * dx), min(float(height), y1 + margin * dy)]
    return tuple(out)


def write_expected_hands(dwpose_path, frame_path, target, cfg):
    if target.exists():
        raise FileExistsError(f"refusing to overwrite expected hand boxes: {target}")
    z = np.load(dwpose_path)
    frames = np.load(frame_path, mmap_mode='r')
    left, right = expected_boxes_from_dwpose(z['hands'], z['hands_score'], width=frames.shape[2], height=frames.shape[1],
        confidence=cfg['t6']['hand_confidence'], minimum_points=cfg['t6']['expected_min_points'], margin=cfg['t6']['expected_box_margin'])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, left_box=left, right_box=right, source=np.asarray('DWPose clean frames; index0=person-left,index1=person-right'),
        confidence=np.asarray(cfg['t6']['hand_confidence']), minimum_points=np.asarray(cfg['t6']['expected_min_points']), margin=np.asarray(cfg['t6']['expected_box_margin']))
    return target


def meta_write(path, clean, edited, edit, expected, extra=None):
    assert_changed_frames(clean,edited,expected)
    d={'edit':edit,'actual_changed_frames':np.flatnonzero(np.any(clean!=edited,axis=(1,2,3))).astype(int).tolist(),
       'expected_changed_frames':list(map(int,expected)),
       'pixel_changed_count':int(np.count_nonzero(clean!=edited)),
       'unchanged_exterior_verified':True}
    if extra:d.update(extra)
    dump(d,path/'edit_meta.json')

def extract_emotion(sample,cfg):
    m=json.loads((INPUT/sample/'input_manifest.json').read_text()); source=m['source']; name=source['raw_video_id']+'.npz'
    with tarfile.open(source['shard']) as t:
        f=t.extractfile(name)
        if f is None:raise FileNotFoundError(name)
        z=np.load(io.BytesIO(f.read()))
        raw=z['movement:emotion_scores'].astype(float)
    aligned,idx=align_emotion_scores(raw,float(source['start_seconds']),cfg['frames'],src_fps=30,dst_fps=24)
    # Official source order is anger/contempt/disgust/fear/happiness/neutral/sadness/surprise;
    # explicit evaluator has no contempt channel, so remove it and renormalize.
    seven=np.delete(aligned,1,axis=1); denom=seven.sum(1,keepdims=True)
    seven=np.divide(seven,denom,out=np.full_like(seven,np.nan),where=denom>0)
    segment=seven.reshape(20,12,7).mean(1)
    valid=np.isfinite(segment).any(axis=1)
    labels=np.full(20,-1,dtype=int)
    labels[valid]=np.argmax(np.nan_to_num(segment[valid],nan=-np.inf),axis=1)
    return labels,idx,valid

def target_spec(labels,cfg):
    names=cfg['emotion']['evaluator_class_order']; step=cfg['emotion']['segment_seconds']
    return {'segments':[{'start':i*step,'end':(i+1)*step,'label':names[int(v)]} for i,v in enumerate(labels) if v >= 0]}

def selected_windows(cfg, requested=None):
    """Return registered windows, optionally a validated nonempty subset."""
    registered = cfg['windows']
    if requested is None:
        return registered
    unknown = [name for name in requested if name not in registered]
    if unknown:
        raise ValueError(f'unregistered window(s): {unknown}')
    if len(set(requested)) != len(requested):
        raise ValueError(f'duplicate window selection: {requested}')
    return {name: registered[name] for name in requested}


def fresh_dir(path):
    """Create an artifact directory once; never overwrite an existing output."""
    if path.exists():
        raise FileExistsError(f'refusing to overwrite existing output directory: {path}')
    path.mkdir(parents=True)
    return path

def prepare_sample(sample,cfg,only_t6=False, windows=None):
    clean_path=decode_once(sample,OUT/'clean'/sample/'clean.npy',cfg); clean=np.load(clean_path)
    audio=clean_audio(sample,cfg); order=cfg['samples']; donor=order[(order.index(sample)+1)%len(order)]
    labels,indices,valid=extract_emotion(sample,cfg); donor_labels, _,donor_valid=extract_emotion(donor,cfg)
    next2_labels, _,next2_valid=extract_emotion(order[(order.index(sample)+2)%len(order)],cfg)
    replacement=np.full(20,-1,dtype=int)
    for i in range(20):
        if valid[i] and donor_valid[i] and donor_labels[i] != labels[i]: replacement[i]=donor_labels[i]
        elif valid[i] and next2_valid[i] and next2_labels[i] != labels[i]: replacement[i]=next2_labels[i]
    for w,(a,b) in selected_windows(cfg, windows).items():
        base=lambda test,level: OUT/test/sample/w/level
        if not only_t6:
            for k in cfg['t1_shifts_24fps']:
                p=base('T1',f'k_{k:+d}'); fresh_dir(p); shift=round(k*cfg['audio_sample_rate']/cfg['fps'])
                changed,meta=shift_audio_window(audio,round(a*cfg['audio_sample_rate']/cfg['fps']),round(b*cfg['audio_sample_rate']/cfg['fps']),shift)
                wavfile.write(p/'audio.wav',cfg['audio_sample_rate'],changed)
                sample_a,sample_b=round(a*cfg['audio_sample_rate']/cfg['fps']),round(b*cfg['audio_sample_rate']/cfg['fps'])
                dump({'edit':meta,'frame_source':str(clean_path),'audio_exterior_unchanged':bool(np.array_equal(audio[:sample_a],changed[:sample_a]) and np.array_equal(audio[sample_b:],changed[sample_b:])),'audio_shift_xcorr':verify_audio_shift_xcorr(audio,changed,sample_a,sample_b,shift)},p/'edit_meta.json')
            donor_frames=np.load(decode_once(donor,OUT/'clean'/donor/'clean.npy',cfg)); x,m=replace_window(clean,donor_frames,a,b);p=base('T2','swap_next');fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,range(a,b),{'donor_sample':donor})
            for name,fun,expect in [('freeze',lambda:freeze_window(clean,a,cfg['t3']['freeze_frames']),range(a+1,a+cfg['t3']['freeze_frames'])),('skip',lambda:skip_forward_window(clean,a,b,cfg['t3']['skip_frames']),range(a,b)),('jump',lambda:jump_frame(clean,a,cfg['t3']['jump_frames']),[a])]:
                x,m=fun();p=base('T3',name);fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,expect)
            for e in cfg['t4_flicker_ev']:
                x,m=apply_flicker_ev(clean,a,b,e);p=base('T4',f'ev_{e:g}');fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,range(a,b))
            x,m=hard_cut_window(clean,a,b,cfg['t4_hard_cut_offset']);p=base('T4','hard_cut');fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,range(a,b))
            for e in cfg['t4_single_frame_ev']:
                x,m=apply_single_frame_exposure_ev(clean,a,e);p=base('T4',f'instant_ev_{e:g}');fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,[a])
            wrong=labels.copy()
            segment_start,segment_end=a//12,b//12
            eligible=np.arange(segment_start,segment_end)[replacement[segment_start:segment_end]>=0]
            wrong[eligible]=replacement[eligible]
            for kind,lab in [('correct',labels),('wrong',wrong)]:
                p=base('T5',kind);fresh_dir(p);dump(target_spec(lab,cfg),p/'target.json');dump({'edit':{'kind':'emotion_target_only','window':[a,b],'source_indices_30fps':indices.tolist(),'label_source':'Imitator automatic movement:emotion_scores','valid_segment_fraction':float(np.mean(lab>=0)),'wrong_target_replaced_segments':list(range(segment_start,segment_end)) if kind=='wrong' else []},'actual_changed_frames':[],'expected_changed_frames':[]},p/'edit_meta.json')
        dw=OUT/'clean'/sample/'evaluators'/'dwpose.npz'
        if dw.exists():
            z=np.load(dw); x,m=occlude_right_hand(clean,z['hands'],z['hands_score'],a,b,right_hand_index=cfg['t6']['right_hand_index'],margin=cfg['t6']['margin']);p=base('T6','right_hand_gray');fresh_dir(p);np.save(p/'frames.npy',x);meta_write(p,clean,x,m,m['applied_frames'],{'skipped_frames':m['skipped_frames']})

def gpu_guard(gpu, peak_mib):
    if str(gpu) == '3': raise ValueError('GPU 3 is permanently prohibited')
    line=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    rows={x.split(',')[0].strip():x.split(',') for x in line.strip().splitlines()}
    free=int(rows[str(gpu)][1]); util=int(rows[str(gpu)][2])
    if free < peak_mib + 4096: raise RuntimeError(f'GPU{gpu} free={free}MiB, need at least {peak_mib+4096}MiB')
    return {'gpu':int(gpu),'free_mib_before':free,'utilization_before':util,'estimated_peak_mib':peak_mib}

def call(cmd, log, env=None, cpu=False):
    log.parent.mkdir(parents=True,exist_ok=True)
    runenv=os.environ.copy(); runenv.update({'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','ORT_THREADS':'8'})
    if env: runenv.update(env)
    prefix=['taskset','-c','0-31'] if cpu else []
    started=time.monotonic()
    with log.open('w') as f: subprocess.run(prefix+list(map(str,cmd)),check=True,stdout=f,stderr=subprocess.STDOUT,env=runenv)
    return time.monotonic()-started

def evaluate(sample, test, window, level, gpu):
    cfg=json.loads((ROOT/'scripts/evaluator_unit_tests/config.json').read_text()); p=OUT/test/sample/window/level
    clean=OUT/'clean'/sample/'clean.npy'; inp=INPUT/sample; out=p/'evaluators'; fresh_dir(out)
    frames=p/'frames.npy' if (p/'frames.npy').exists() else clean
    audio=p/'audio.wav' if (p/'audio.wav').exists() else inp/'audio.wav'
    target=p/'target.json' if (p/'target.json').exists() else OUT/'T5'/sample/window/'correct'/'target.json'
    e=ROOT/'scripts/evaluators'; acc='/data/lanyi/.conda/envs/cfpr-v2-acc/bin/python'; dh='/data/lanyi/.conda/envs/dh-eval-prm/bin/python'; sync='/data/lanyi/.conda/envs/syncnet/bin/python'; hand='/data/lanyi/.conda/envs/cfpr-wilor-evaluator/bin/python'
    timing={}
    timing['dwpose_cpu_seconds']=call([acc,e/'extract_dwpose.py',frames,out/'dwpose.npz'],out/'log_dwpose.txt',cpu=True)
    timing['arcface_cpu_seconds']=call([dh,e/'score_emotion_refs.py',out/'face.json',f'gen={inp}/reference.png,{frames}'],out/'log_face.txt',cpu=True)
    timing['lipsync_cpu_seconds']=call([sync,e/'eval_lipsync.py','gen',frames,audio,out/'lipsync'],out/'log_lipsync.txt',cpu=True)
    guard=[]; guard.append(gpu_guard(gpu,4096)); timing['dino_gpu_seconds']=call([acc,e/'eval_identity_dino.py',out/'dino.json',sample,f'gen={frames}'],out/'log_dino.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    guard.append(gpu_guard(gpu,4096)); timing['temporal_evidence_gpu_seconds']=call([dh,e/'eval_temporal_evidence.py',frames,out/'temporal'],out/'log_temporal.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    timing['temporal_context_cpu_seconds']=call([acc,e/'eval_temporal_context.py',out/'temporal'],out/'log_temporal_context.txt',cpu=True)
    timing['pose_cpu_seconds']=call([acc,e/'eval_pose.py',out/'dwpose.npz',out/'pose','--real',OUT/'clean'/sample/'evaluators'/'dwpose.npz'],out/'log_pose.txt',cpu=True)
    timing['emotion_cpu_seconds']=call([acc,e/'eval_emotion_explicit.py',out/'face.json',target,out/'emotion.json'],out/'log_emotion.txt',cpu=True)
    expected=OUT/'clean'/sample/'expected_hands.npz'
    if not expected.exists(): raise FileNotFoundError(f'expected hand boxes required before condition evaluation: {expected}')
    guard.append(gpu_guard(gpu,6144)); timing['hand_gpu_seconds']=call([hand,e/'eval_hand_joint_visibility.py',frames,out/'hand_joint_visibility.npz','--stride','1','--expected',expected,'--expected-iou',str(cfg['t6']['expected_iou'])],out/'log_hand.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    dump({'sample':sample,'test':test,'window':window,'level':level,'frame_source':str(frames),'audio_source':str(audio),'target_source':str(target),'gpu_preflight':guard,'wall_clock_seconds':timing},out/'run_meta.json')


def evaluate_clean(sample, gpu):
    cfg=json.loads((ROOT/'scripts/evaluator_unit_tests/config.json').read_text())
    p=OUT/'clean'/sample
    clean=decode_once(sample,p/'clean.npy',cfg); inp=INPUT/sample; out=p/'evaluators'; fresh_dir(out)
    target=OUT/'T5'/sample/'W1'/'correct'/'target.json'
    if not target.exists(): raise FileNotFoundError('prepare_sample must create the frozen T5 target before clean evaluation')
    e=ROOT/'scripts/evaluators'; acc='/data/lanyi/.conda/envs/cfpr-v2-acc/bin/python'; dh='/data/lanyi/.conda/envs/dh-eval-prm/bin/python'; sync='/data/lanyi/.conda/envs/syncnet/bin/python'; hand='/data/lanyi/.conda/envs/cfpr-wilor-evaluator/bin/python'
    timing={}
    timing['dwpose_cpu_seconds']=call([acc,e/'extract_dwpose.py',clean,out/'dwpose.npz'],out/'log_dwpose.txt',cpu=True)
    expected=write_expected_hands(out/'dwpose.npz',clean,p/'expected_hands.npz',cfg)
    timing['arcface_cpu_seconds']=call([dh,e/'score_emotion_refs.py',out/'face.json',f'gen={inp}/reference.png,{clean}'],out/'log_face.txt',cpu=True)
    timing['lipsync_cpu_seconds']=call([sync,e/'eval_lipsync.py','gen',clean,inp/'audio.wav',out/'lipsync'],out/'log_lipsync.txt',cpu=True)
    guard=[]; guard.append(gpu_guard(gpu,4096)); timing['dino_gpu_seconds']=call([acc,e/'eval_identity_dino.py',out/'dino.json',sample,f'gen={clean}'],out/'log_dino.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    guard.append(gpu_guard(gpu,4096)); timing['temporal_evidence_gpu_seconds']=call([dh,e/'eval_temporal_evidence.py',clean,out/'temporal'],out/'log_temporal.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    timing['temporal_context_cpu_seconds']=call([acc,e/'eval_temporal_context.py',out/'temporal'],out/'log_temporal_context.txt',cpu=True)
    timing['pose_cpu_seconds']=call([acc,e/'eval_pose.py',out/'dwpose.npz',out/'pose','--real',out/'dwpose.npz'],out/'log_pose.txt',cpu=True)
    timing['emotion_cpu_seconds']=call([acc,e/'eval_emotion_explicit.py',out/'face.json',target,out/'emotion.json'],out/'log_emotion.txt',cpu=True)
    guard.append(gpu_guard(gpu,6144)); timing['hand_gpu_seconds']=call([hand,e/'eval_hand_joint_visibility.py',clean,out/'hand_joint_visibility.npz','--stride','1','--expected',expected,'--expected-iou',str(cfg['t6']['expected_iou'])],out/'log_hand.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    dump({'sample':sample,'mode':'clean_baseline_v2','frame_source':str(clean),'audio_source':str(inp/'audio.wav'),'target_source':str(target),'expected_hands':str(expected),'gpu_preflight':guard,'wall_clock_seconds':timing},out/'run_meta.json')

def evaluate_clean_repeat(sample, tag, gpu):
    """Run the fixed clean input into a fresh, immutable determinism directory."""
    cfg=json.loads((ROOT/'scripts/evaluator_unit_tests/config.json').read_text())
    p=OUT/'clean_repeats'/sample/tag
    if p.exists():
        raise FileExistsError(f'refusing to overwrite clean repeat output: {p}')
    inp=INPUT/sample; clean=OUT/'clean'/sample/'clean.npy'; out=p/'evaluators'; fresh_dir(out)
    target=OUT/'T5'/sample/'W1'/'correct'/'target.json'
    if not clean.exists() or not target.exists():
        raise FileNotFoundError('clean baseline and its registered W1 emotion target must exist first')
    e=ROOT/'scripts/evaluators'; acc='/data/lanyi/.conda/envs/cfpr-v2-acc/bin/python'; dh='/data/lanyi/.conda/envs/dh-eval-prm/bin/python'; sync='/data/lanyi/.conda/envs/syncnet/bin/python'; hand='/data/lanyi/.conda/envs/cfpr-wilor-evaluator/bin/python'
    timing={}
    timing['dwpose_cpu_seconds']=call([acc,e/'extract_dwpose.py',clean,out/'dwpose.npz'],out/'log_dwpose.txt',cpu=True)
    timing['arcface_cpu_seconds']=call([dh,e/'score_emotion_refs.py',out/'face.json',f'gen={inp}/reference.png,{clean}'],out/'log_face.txt',cpu=True)
    timing['lipsync_cpu_seconds']=call([sync,e/'eval_lipsync.py','gen',clean,inp/'audio.wav',out/'lipsync'],out/'log_lipsync.txt',cpu=True)
    guard=[]; guard.append(gpu_guard(gpu,4096)); timing['dino_gpu_seconds']=call([acc,e/'eval_identity_dino.py',out/'dino.json',sample,f'gen={clean}'],out/'log_dino.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    guard.append(gpu_guard(gpu,4096)); timing['temporal_evidence_gpu_seconds']=call([dh,e/'eval_temporal_evidence.py',clean,out/'temporal'],out/'log_temporal.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    timing['temporal_context_cpu_seconds']=call([acc,e/'eval_temporal_context.py',out/'temporal'],out/'log_temporal_context.txt',cpu=True)
    timing['pose_cpu_seconds']=call([acc,e/'eval_pose.py',out/'dwpose.npz',out/'pose','--real',OUT/'clean'/sample/'evaluators'/'dwpose.npz'],out/'log_pose.txt',cpu=True)
    timing['emotion_cpu_seconds']=call([acc,e/'eval_emotion_explicit.py',out/'face.json',target,out/'emotion.json'],out/'log_emotion.txt',cpu=True)
    expected=OUT/'clean'/sample/'expected_hands.npz'
    if not expected.exists(): raise FileNotFoundError(f'expected hand boxes required before determinism repeat: {expected}')
    guard.append(gpu_guard(gpu,6144)); timing['hand_gpu_seconds']=call([hand,e/'eval_hand_joint_visibility.py',clean,out/'hand_joint_visibility.npz','--stride','1','--expected',expected,'--expected-iou',str(cfg['t6']['expected_iou'])],out/'log_hand.txt',{'CUDA_VISIBLE_DEVICES':str(gpu)})
    dump({'sample':sample,'mode':'clean_determinism_repeat','tag':tag,'frame_source':str(clean),'audio_source':str(inp/'audio.wav'),'target_source':str(target),'gpu_preflight':guard,'wall_clock_seconds':timing},out/'run_meta.json')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'scripts/evaluator_unit_tests/config.json'));ap.add_argument('--sample',action='append');ap.add_argument('--window',action='append', help='prepare only this registered window; refuses existing condition directories');ap.add_argument('--only-t6',action='store_true');ap.add_argument('--evaluate',nargs=4,metavar=('TEST','WINDOW','LEVEL','GPU'));ap.add_argument('--evaluate-clean',metavar='GPU');ap.add_argument('--validate-t1-audio',nargs=2,metavar=('WINDOW','LEVEL'));ap.add_argument('--clean-repeat',nargs=2,metavar=('TAG','GPU'));a=ap.parse_args();cfg=json.loads(Path(a.config).read_text())
    if a.validate_t1_audio:
        if not a.sample or len(a.sample)!=1: raise ValueError('--validate-t1-audio needs exactly one --sample')
        validate_existing_t1_audio(a.sample[0],*a.validate_t1_audio,cfg); return
    if a.evaluate_clean:
        if not a.sample or len(a.sample)!=1: raise ValueError('--evaluate-clean needs exactly one --sample')
        evaluate_clean(a.sample[0],a.evaluate_clean); return
    if a.clean_repeat:
        if not a.sample or len(a.sample)!=1: raise ValueError('--clean-repeat needs exactly one --sample')
        evaluate_clean_repeat(a.sample[0],*a.clean_repeat); return
    if a.evaluate:
        if not a.sample or len(a.sample)!=1: raise ValueError('--evaluate needs exactly one --sample')
        evaluate(a.sample[0],*a.evaluate); return
    for s in (a.sample or cfg['samples']):prepare_sample(s,cfg,a.only_t6,a.window)

if __name__=='__main__':main()
