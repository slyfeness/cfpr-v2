import json
from pathlib import Path
import cv2, numpy as np
R=Path('/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/run_20261110')
I=Path('/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1')
M=json.loads((R/'pilot_measurements.json').read_text())
out=R/'contact_sheets';out.mkdir(exist_ok=True)
def crop(f,b):
 if b is None:return np.zeros((160,160,3),np.uint8)
 x0,y0,x1,y1=b; d=int(.15*max(x1-x0,y1-y0)); x0=max(0,x0-d);y0=max(0,y0-d);x1=min(f.shape[1],x1+d);y1=min(f.shape[0],y1+d)
 return cv2.resize(f[y0:y1,x0:x1],(160,160),interpolation=cv2.INTER_CUBIC)
for s,d in [('CRB100_001','CRB100_002'),('CRB100_025','CRB100_050')]:
 q=M['samples'][s]; clean=np.load(R/s/'clean_a/frames_uint8.npy'); donor=cv2.cvtColor(cv2.imread(str(I/d/'reference.png')),cv2.COLOR_BGR2RGB)
 # donor reference is display-only; frames use their per-frame detector boxes.
 donor_tile=cv2.resize(donor,(160,160),interpolation=cv2.INTER_CUBIC)
 rows=[]
 for p in [f'P{i}' for i in range(8)]:
  v=np.load(R/s/p/'frames_uint8.npy'); tiles=[]
  for f in [108,120,132]:tiles += [crop(clean[f],q['clean']['boxes'][f]),crop(v[f],q['variants'][p]['boxes'][f])]
  tiles += [donor_tile]
  row=np.concatenate(tiles,axis=1);cv2.putText(row,p,(5,22),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,0,0),2);rows.append(row)
 sheet=np.concatenate(rows,axis=0);cv2.imwrite(str(out/f'{s}_pilot_faces.png'),cv2.cvtColor(sheet,cv2.COLOR_RGB2BGR))
