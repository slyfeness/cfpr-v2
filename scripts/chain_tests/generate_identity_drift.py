#!/usr/bin/env python3
"""Direct shared-GPU-safe generator for the frozen identity-drift matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
from PIL import Image
from scripts.chain_tests.identity_config import build_tasks
from scripts.chain_tests.identity_edit import identity_state_record, make_identity_ctx_edit
from src import acc_core
from src.models.mutual_self_attention import ReferenceAttentionControl, torch_dfs

INPUT=Path('/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1')
DEFAULT_ROOT=Path('/data/lanyi/outputs/cfpr-v2/chain_identity_drift/run_20261109')

def reader_modules(pipe):
    return sorted([m for m in torch_dfs(pipe.denoising_unet) if type(m).__name__=='TemporalBasicTransformerBlock'],key=lambda x:-x.norm1.normalized_shape[0])

@torch.no_grad()
def bank_of(pipe,img):
    writer=ReferenceAttentionControl(pipe.reference_unet,do_classifier_free_guidance=False,mode='write',batch_size=1,fusion_blocks='full')
    x=pipe.ref_image_processor.preprocess(img,height=acc_core.H,width=acc_core.W).to(dtype=pipe.vae.dtype,device='cuda')
    pipe.reference_unet(pipe.vae.encode(x).latent_dist.mean*.18215,torch.zeros(1,device='cuda',dtype=torch.long),encoder_hidden_states=None,return_dict=False)
    bank=[m.bank[0].clone() for m in acc_core.writer_modules(pipe)]; writer.clear(); return bank

def run(pipe, root, aligned, manifest, task):
    d=root/task['out_id'];
    if d.exists(): raise FileExistsError(d)
    torch.cuda.reset_peak_memory_stats()
    ref,audio,poses=acc_core.load_inputs(INPUT/task['sample'],task['frames']); edit=None
    state=None
    meta={'actions':[], 'step0_contexts':[], 'self_check':{'not_applicable_clean_generation':True}}
    if task['actions']:
        action=task['actions'][0]; donor=action['donor']; db=bank_of(pipe,Image.open(aligned/f'{task["sample"]}__donor_{donor}_aligned.png').convert('RGB'))
        edit,state=make_identity_ctx_edit(reader_modules(pipe),db,manifest[task['sample']]['own_face_box'],action)
        meta={'actions':[action],'own_face_box':manifest[task['sample']]['own_face_box'],'donor_face_box':manifest[task['sample']]['donor_face_box']}
    vid,info,steps=acc_core.generate(pipe,ref,audio,poses,task['frames'],seed=task['seed'],ctx_edit=edit)
    info['chain_identity_drift_task']=task
    if state is not None:
        checks=state['checks']
        required=(checks['zero_strength_elementwise'] and checks['zero_strength_original_object'] and checks['outside_face_identical'] and checks['inside_formula_within_fp16_rounding'])
        if not required:
            raise RuntimeError(f"identity edit self-check failed for {task['out_id']}: {checks}")
        meta['per_layer_participating_token_counts']=state['token_counts']
        meta['step0_contexts']=state['step0_contexts']
        meta['self_check']=checks
    acc_core.save(d,vid,info,audio,final_latents=steps[-1])
    (d/'edit_meta.json').write_text(json.dumps(meta,indent=2))
    (d/'state_record.json').write_text(json.dumps(identity_state_record(task,meta['step0_contexts']),indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out-root',type=Path,default=DEFAULT_ROOT); p.add_argument('--samples',nargs='*'); p.add_argument('--skip-existing',action='store_true'); a=p.parse_args(); aligned=a.out_root/'alignment'
    manifest=json.loads((aligned/'alignment_manifest.json').read_text()); tasks=build_tasks(); tasks=[t for t in tasks if not a.samples or t['sample'] in a.samples]
    if a.skip_existing:
        tasks=[t for t in tasks if not (a.out_root/t['out_id']).exists()]
    if not tasks:
        print(json.dumps({'complete':0,'total':0,'status':'no_missing_registered_tasks'})); return
    pipe=acc_core.load_pipe()
    try:
        for i,t in enumerate(tasks,1): run(pipe,a.out_root,aligned,manifest,t); print(json.dumps({'complete':i,'total':len(tasks),'task':t['out_id']}),flush=True)
    finally: del pipe; torch.cuda.empty_cache()
if __name__=='__main__': main()
