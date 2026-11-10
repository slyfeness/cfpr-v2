"""Post-hoc sharpness v2: 256px detected-face Laplacian, donor-clean denominator."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import cv2,numpy as np
from insightface.app import FaceAnalysis

WINDOW=(96,144)
PAIRS={'CRB100_001':'CRB100_002','CRB100_025':'CRB100_050'}
STEP3=Path('/data/lanyi/outputs/cfpr-v2/chain_identity_drift/run_20261109')

def laplacian256(crop):
    if crop is None or not crop.size:return float('nan')
    crop=cv2.resize(crop,(256,256),interpolation=cv2.INTER_CUBIC)
    gray=crop if crop.ndim==2 else cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray,cv2.CV_64F).var())

def v2_ratios(value,donor_clean,own_clean):
    m=lambda x: float(np.nanmean(x)) if np.isfinite(x).any() else float('nan')
    return {'donor_clean_ratio':m(value)/m(donor_clean),'own_clean_ratio':m(value)/m(own_clean)}

def series(frames,detector):
    out=[]; boxes=[]
    for f in frames:
        faces=detector.get(cv2.cvtColor(f,cv2.COLOR_RGB2BGR))
        if not faces:out.append(float('nan'));boxes.append(None);continue
        z=max(faces,key=lambda x:(x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]));x0,y0,x1,y1=np.round(z.bbox).astype(int);x0,y0=max(0,x0),max(0,y0);x1,y1=min(f.shape[1],x1),min(f.shape[0],y1)
        out.append(laplacian256(f[y0:y1,x0:x1]));boxes.append([int(x0),int(y0),int(x1),int(y1)])
    return np.asarray(out),boxes

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variants',nargs='+',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 d=FaceAnalysis(name='buffalo_l',root='/data/lanyi/tmp/insightface_root',providers=['CPUExecutionProvider'],allowed_modules=['detection']);d.prepare(ctx_id=-1,det_size=(640,640)); result={'definition':'detected face crop -> 256x256 bicubic -> grayscale Laplacian variance; donor clean denominator','window':list(WINDOW),'samples':{}}
 for own,donor in PAIRS.items():
  ownv=np.load(a.root/own/'clean_a/frames_uint8.npy');donv=np.load(STEP3/donor/'clean_a/frames_uint8.npy');os,_=series(ownv,d);ds,_=series(donv,d);rows={}
  for v in a.variants:
   fr=np.load(a.root/own/v/'frames_uint8.npy'); s,b=series(fr,d); rows[v]={'sharpness_256':s.tolist(),'boxes':b,**v2_ratios(s[WINDOW[0]:WINDOW[1]],ds[WINDOW[0]:WINDOW[1]],os[WINDOW[0]:WINDOW[1]]),'valid_face_frames':int(np.isfinite(s).sum())}
  result['samples'][own]={'own_clean_256':os.tolist(),'donor_clean_sample':donor,'donor_clean_256':ds.tolist(),'variants':rows}
 a.out.write_text(json.dumps(result,indent=2,allow_nan=True))
if __name__=='__main__':main()
