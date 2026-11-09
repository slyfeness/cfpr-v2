#!/usr/bin/env python3
"""Render audit-only side-by-side previews; never feeds an evaluator."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path('/data/lanyi/projects/cfpr-v2')
OUT=Path('/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2')
INPUT=Path('/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1')

def preview_frame_ids(n, start, end, context=24):
    return list(range(max(0, start-context), min(n, end+context)))


def resolve_ffmpeg(explicit=None):
    candidates=[explicit, os.environ.get('CFPR_FFMPEG'), '/data/lanyi/.conda/envs/cfpr-v2-acc/bin/ffmpeg', shutil.which('ffmpeg')]
    for candidate in candidates:
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    raise FileNotFoundError('No existing ffmpeg binary found; do not install one during this audit')

def _panel(frame, label, frame_id):
    image=Image.fromarray(np.asarray(frame)).resize((384,384),Image.Resampling.BICUBIC)
    draw=ImageDraw.Draw(image); draw.rectangle((0,0,384,25),fill=(0,0,0)); draw.text((5,5),f'{label} / frame {frame_id}',fill='white')
    return np.asarray(image)

def _joined(clean, edited, frame_id, title):
    left=_panel(clean[frame_id],'clean',frame_id); right=_panel(edited[frame_id],title,frame_id)
    return np.concatenate([left,right],axis=1)

def render(sample,test,window,level):
    cfg=json.loads((ROOT/'scripts/evaluator_unit_tests/config.json').read_text()); a,b=cfg['windows'][window]
    clean=np.load(OUT/'clean'/sample/'clean.npy',mmap_mode='r'); base=OUT/test/sample/window/level
    edited=np.load(base/'frames.npy',mmap_mode='r') if (base/'frames.npy').exists() else clean
    if len(clean)!=len(edited):raise ValueError('preview input frame length mismatch')
    target=OUT/'previews'/sample/window;target.mkdir(parents=True,exist_ok=True)
    stem=f'{test}__{level}'; title=f'{test} / {sample} / {window} / {level}'
    ids=preview_frame_ids(len(clean),a,b)
    contact_ids=np.linspace(a,b-1,6).round().astype(int).tolist()
    tiles=[np.asarray(Image.fromarray(_joined(clean,edited,i,title)).resize((320,160),Image.Resampling.BICUBIC)) for i in contact_ids]
    contact=np.concatenate([np.concatenate(tiles[:3],axis=1),np.concatenate(tiles[3:],axis=1)],axis=0)
    Image.fromarray(contact).save(target/(stem+'.png'),optimize=True,compress_level=9)
    command=[resolve_ffmpeg(),'-y','-f','rawvideo','-pixel_format','rgb24','-video_size','768x384','-framerate',str(cfg['fps']),' -i'.strip(),'-']
    if test=='T1':
        audio=base/'audio.wav'; command += ['-ss',str(ids[0]/cfg['fps']),'-t',str(len(ids)/cfg['fps']),'-i',str(audio)]
    command += ['-c:v','libx264','-crf','28','-pix_fmt','yuv420p']
    if test=='T1':command += ['-c:a','aac','-shortest']
    else:command += ['-an']
    command += [str(target/(stem+'.mp4'))]
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for frame_id in ids:process.stdin.write(_joined(clean,edited,frame_id,title).tobytes())
    process.stdin.close(); stderr=process.stderr.read(); process.wait()
    if process.returncode:raise RuntimeError(stderr.decode(errors='replace'))
    return {'video':str(target/(stem+'.mp4')),'contact_sheet':str(target/(stem+'.png')),'frames':ids,'contact_frames':contact_ids}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sample',required=True);parser.add_argument('--test',required=True);parser.add_argument('--window',required=True);parser.add_argument('--level',required=True);args=parser.parse_args()
    print(json.dumps(render(args.sample,args.test,args.window,args.level),indent=2))
if __name__=='__main__':main()
