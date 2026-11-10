#!/usr/bin/env python3
"""Direct round-2 generator; uses only ACC's existing ctx_edit hook."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import cv2, torch
from PIL import Image
from scripts.chain_tests.identity_pilot_round2 import expand_box, make_round2_ctx_edit, round2_specs
from scripts.chain_tests.generate_identity_pilot import INPUT, STEP3, PAIRS, SEED, FRAMES, bank_of, reader_modules
from src import acc_core

DEFAULT_ROOT=Path('/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/round2_20261110')

def link_clean(root,sample):
 src=STEP3/sample/'clean_a'; record=json.loads((src/'summary.json').read_text())['raw_output_sha256']
 dst=root/sample/'clean_a';dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=src.resolve(): raise FileExistsError(dst)
 else: dst.symlink_to(src,target_is_directory=True)
 return {'source':str(src),'raw_output_sha256':record,'verified_against_step3_record':True}

def h50_check(root,sample,donor,image,box):
 h,w=image.shape[:2]; x0,y0,x1,y1=[int(v*s) for v,s in zip(box,(w,h,w,h))]
 out=image.copy();cv2.rectangle(out,(x0,y0),(x1,y1),(0,255,0),3);cv2.putText(out,'H50 expanded face box',(12,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
 cv2.imwrite(str(root/sample/'H50_box_check.png'),out)

def run(pipe,root,sample,donor,alignment,spec):
 out=root/sample/spec['id']
 if out.exists(): raise FileExistsError(out)
 own_ref=INPUT/sample/'reference.png'; aligned=STEP3/'alignment'/f'{sample}__donor_{donor}_aligned.png'
 box=expand_box(alignment[sample]['own_face_box'],float(spec['pad']))
 if spec['id']=='H50':h50_check(root,sample,donor,cv2.imread(str(own_ref)),box)
 ref,audio,poses=acc_core.load_inputs(INPUT/sample,FRAMES)
 donor_bank=bank_of(pipe,Image.open(aligned).convert('RGB'))
 action=dict(spec,donor=donor,face_box=box)
 edit,state=make_round2_ctx_edit(reader_modules(pipe),donor_bank,box,action)
 torch.cuda.reset_peak_memory_stats(); vid,info,steps=acc_core.generate(pipe,ref,audio,poses,FRAMES,seed=SEED,ctx_edit=edit)
 info['identity_injection_pilot_round2']={'sample':sample,'donor':donor,'variant':spec,'clean_reused':str(root/sample/'clean_a')}
 acc_core.save(out,vid,info,audio,final_latents=steps[-1])
 (out/'edit_meta.json').write_text(json.dumps({'sample':sample,'donor':donor,'variant':spec,'face_box':box,'contexts':state['contexts'],'selected_face_token_counts':state['selected_face_token_counts']},indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('--out-root',type=Path,default=DEFAULT_ROOT);p.add_argument('--samples',nargs='*');p.add_argument('--variants',nargs='*');a=p.parse_args()
 samples=[s for s in PAIRS if not a.samples or s in a.samples];specs=[x for x in round2_specs() if not a.variants or x['id'] in a.variants]
 a.out_root.mkdir(parents=True,exist_ok=True); alignment=json.loads((STEP3/'alignment'/'alignment_manifest.json').read_text());(a.out_root/'clean_reuse_audit.json').write_text(json.dumps({s:link_clean(a.out_root,s) for s in samples},indent=2))
 pipe=acc_core.load_pipe()
 try:
  for s in samples:
   for spec in specs: run(pipe,a.out_root,s,PAIRS[s],alignment,spec);print(json.dumps({'sample':s,'variant':spec['id'],'status':'complete'}),flush=True)
 finally: del pipe;torch.cuda.empty_cache()
if __name__=='__main__':main()
