#!/usr/bin/env python3
"""Create registered donor alignments and human-verifiable face-box sheets."""
import argparse, json
from pathlib import Path
import cv2, numpy as np
from scripts.chain_tests.identity_edit import expanded_face_box

SAMPLES=("CRB100_001","CRB100_002","CRB100_025","CRB100_050","CRB100_056","CRB100_075","CRB100_087","CRB100_090")
DONOR={sample:SAMPLES[(i+1)%len(SAMPLES)] for i,sample in enumerate(SAMPLES)}

def face_box(input_root, sample):
    pose=np.load(Path(input_root)/sample/"pose/0.npy",allow_pickle=True).tolist()
    return expanded_face_box(np.asarray(pose["faces"])[0],np.asarray(pose["faces_score"])[0])

def align(donor, donor_box, own_box):
    h,w=donor.shape[:2]; dcx=(donor_box[0]+donor_box[2])*w/2; dcy=(donor_box[1]+donor_box[3])*h/2
    ocx=(own_box[0]+own_box[2])*w/2; ocy=(own_box[1]+own_box[3])*h/2
    scale=((own_box[3]-own_box[1])/(donor_box[3]-donor_box[1]))
    mat=np.array([[scale,0,ocx-scale*dcx],[0,scale,ocy-scale*dcy]],np.float32)
    return cv2.warpAffine(donor,mat,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE),mat

def draw(im, box, text):
    out=im.copy(); h,w=out.shape[:2]; x0,y0,x1,y1=(int(box[0]*w),int(box[1]*h),int(box[2]*w),int(box[3]*h))
    cv2.rectangle(out,(x0,y0),(x1,y1),(0,255,0),3); cv2.putText(out,text,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--input-root",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); a=ap.parse_args(); a.out.mkdir(parents=True,exist_ok=False)
    rows={}
    for own,donor in DONOR.items():
        own_box=face_box(a.input_root,own); donor_box=face_box(a.input_root,donor)
        own_img=cv2.imread(str(a.input_root/own/"reference.png")); donor_img=cv2.imread(str(a.input_root/donor/"reference.png"))
        if own_img is None or donor_img is None: raise FileNotFoundError(f"reference missing: {own}/{donor}")
        aligned,mat=align(donor_img,donor_box,own_box)
        cv2.imwrite(str(a.out/f"{own}__donor_{donor}_aligned.png"),aligned)
        sheet=np.concatenate([draw(own_img,own_box,f"own {own}"),draw(aligned,own_box,f"aligned donor {donor}")],axis=1)
        cv2.imwrite(str(a.out/f"{own}__alignment_check.png"),sheet)
        rows[own]={"donor":donor,"own_face_box":own_box,"donor_face_box":donor_box,"affine_scale_translate":mat.tolist()}
    (a.out/"alignment_manifest.json").write_text(json.dumps(rows,indent=2))
if __name__=="__main__": main()
