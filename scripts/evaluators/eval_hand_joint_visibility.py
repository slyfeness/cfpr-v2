"""Native per-joint visibility for RGB frames; expected-hand matching is additive."""
from pathlib import Path
import argparse,sys,json,os,numpy as np
R=Path('/data/lanyi/projects/cfpr-v2');sys.path.insert(0,str(R))
from src.hand_joint_visibility import DetectedHandVisibility
from src.hand_expected_visibility import associate_expected_hands

def load_expected(path, n):
 if path is None:return None
 z=np.load(path)
 out={}
 for key in ('left_box','right_box'):
  value=np.asarray(z[key],float)
  if value.shape != (n,4):raise ValueError(f'Expected {key} shape {(n,4)}, got {value.shape}')
  out[key]=value
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('frames');p.add_argument('output');p.add_argument('--stride',type=int,default=1);p.add_argument('--expected');p.add_argument('--expected-iou',type=float,default=.3);a=p.parse_args()
 visible=os.environ.get('CUDA_VISIBLE_DEVICES')
 if not visible or '3' in {x.strip() for x in visible.split(',')}:raise ValueError('Hand visibility requires one explicit non-GPU3 CUDA device')
 if a.stride<1:raise ValueError('Stride must be positive')
 if not 0 <= a.expected_iou <= 1:raise ValueError('expected IoU must be in [0,1]')
 frames=np.load(a.frames,mmap_mode='r')
 if frames.dtype!=np.uint8 or frames.ndim!=4 or frames.shape[-1]!=3:raise ValueError('Expected uint8 RGB THWC frame array')
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
 if out.is_symlink() or out.with_suffix('.json').is_symlink():raise ValueError('Refusing linked output')
 n=len(frames); expected=load_expected(a.expected,n)
 prob=np.full(n,np.nan,np.float32);frac=np.full(n,np.nan,np.float32);count=np.full(n,np.nan,np.float32);sampled=np.zeros(n,bool)
 vis_left=np.full(n,np.nan,np.float32);vis_right=np.full(n,np.nan,np.float32);missing_left=np.full(n,np.nan,np.float32);missing_right=np.full(n,np.nan,np.float32)
 records=[];region_frames=[];region_boxes=[];region_prob=[];region_side=[];region_conf=[]
 m=DetectedHandVisibility()
 for i in range(0,n,a.stride):
  z=m.predict_image(frames[i]);sampled[i]=True;count[i]=len(z['regions'])
  if z['state']=='Measured':prob[i]=z['mean_joint_visibility'];frac[i]=z['visible_joint_fraction']
  records.append(dict(frame=i,**z))
  for region in z['regions']:
   region_frames.append(i);region_boxes.append(region['box']);region_prob.append(region['visibility']);region_side.append(region['canonical_is_right']);region_conf.append(region['detector_confidence'])
  if expected is not None:
   matched=associate_expected_hands(z['regions'],expected['left_box'][i],expected['right_box'][i],a.expected_iou)
   vis_left[i]=matched['vis_left'];vis_right[i]=matched['vis_right'];missing_left[i]=matched['missing_left'];missing_right[i]=matched['missing_right']
  if i%40==0:print('HAND_VISIBILITY_FRAME',i,n,z['state'],flush=True)
 np.savez_compressed(out,evaluator_id=np.asarray('hand_visibility_detector_wilor_frozen_6321d62f'),mean_joint_visibility=prob,visible_joint_fraction=frac,detected_hand_count=count,sampled=sampled,stride=np.asarray(a.stride),region_frame=np.asarray(region_frames,int),region_box=np.asarray(region_boxes,np.float32).reshape(-1,4),region_joint_visibility=np.asarray(region_prob,np.float32).reshape(-1,21),region_is_right=np.asarray(region_side,bool),region_detector_confidence=np.asarray(region_conf,np.float32),vis_left=vis_left,vis_right=vis_right,missing_left=missing_left,missing_right=missing_right)
 meta=dict(evaluator_id='hand_visibility_detector_wilor_frozen_6321d62f',frames=n,stride=a.stride,sampled_frames=int(sampled.sum()),measured_frames=int(np.isfinite(prob).sum()),regions=len(region_frames),frame_mean_visibility=float(np.nanmean(prob)) if np.isfinite(prob).any() else None,frame_mean_visible_fraction=float(np.nanmean(frac)) if np.isfinite(frac).any() else None,scope='Native21joint visibility only, no finger-count/anatomy judgment. Aggregate over all detected hand regions, no identity association.',missingness='No native detected hand means Unknown/NaN, not zero or anatomical Normal. Unsampled frames remain NaN. No interpolation.',expected_hand_mode=None if expected is None else dict(source=str(a.expected),iou=float(a.expected_iou),semantics='Expected detected hand matched by same side and IoU; expected-but-unmatched has visibility 0 and missing 1; not-expected remains NaN.'),source_revision='6321d62fb7617cf504d1de6e4d9ecda7aadfb989',head_sha256='c26712319e82fd5701d4eb1dd597d69469b770140e5fbf59252c6f08863b44cb',native_preprocessing='RGB square256 expanded1.25; native left-hand flip; ImageNet norm; center192wide',validation='64hash-selected HInt officialTEST, automatic61/64, jointF1.8867; 2existingACC clips16frames gray-hand occlusion median visibility drop.4433, all16covered frames Unknown; not generated anatomical accuracy',records=records)
 out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
 print('HAND_VISIBILITY_COMPLETE',meta['measured_frames'],n,flush=True)
if __name__=='__main__':main()
