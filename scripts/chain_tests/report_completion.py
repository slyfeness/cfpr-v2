#!/usr/bin/env python3
"""Read-only completion report for the frozen chain lip-offset run."""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path

SAMPLES = ("CRB100_001","CRB100_002","CRB100_025","CRB100_050","CRB100_056","CRB100_075","CRB100_087","CRB100_090")
WINDOW_SEGMENTS={"W1":range(8,12),"W2":range(14,18)}
REGISTERED_K=(-8,-4,4,8)

def g4_person_status(windows: dict) -> str:
    values=[windows[w][str(k)]["G4_pass"] for w in ("W1","W2") for k in REGISTERED_K]
    if any(value is None for value in values): return "undetermined"
    return "pass" if all(values) else "fail"

def load(path): return json.loads(Path(path).read_text())
def med(values): return None if not values else statistics.median(values)

def required(root: Path, receipt: dict):
    missing=[]
    for sample in SAMPLES:
        for suffix in ("clean_a/evaluators/face.json","clean_a/evaluators/pose/pose.json","clean_a/evaluators/temporal/gen_temporal_context.json"):
            if not (root/sample/suffix).exists(): missing.append(str(root/sample/suffix))
        for window in ("W1","W2"):
            for k in (-8,-4,-2,2,4,8):
                base=root/sample/window/f"k_{k:+d}"/"evaluators"
                for suffix in ("face.json","pose/pose.json","temporal/gen_temporal_context.json"):
                    if not (base/suffix).exists(): missing.append(str(base/suffix))
        hp=Path("/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2/clean")/sample/"evaluators/lipsync/gen_lipsync.json"
        if not hp.exists(): missing.append(str(hp))
    return missing

def task_side(root, sample, window, k):
    base=root/sample/"clean_a"/"evaluators"; edit=root/sample/window/f"k_{k:+d}"/"evaluators"
    f0,f1=load(base/"face.json")["gen"],load(edit/"face.json")["gen"]
    p0,p1=load(base/"pose/pose.json")["gen"],load(edit/"pose/pose.json")["gen"]
    t0,t1=load(base/"temporal/gen_temporal_context.json"),load(edit/"temporal/gen_temporal_context.json")
    return {"arcface_delta":f1["arcface_gen_vs_orig_ref"]-f0["arcface_gen_vs_orig_ref"],"valence_delta":f1["gen_valence_mean"]-f0["gen_valence_mean"],"happy_delta":f1["gen_p_happy_mean"]-f0["gen_p_happy_mean"],"pose_flags_delta":p1["any_flag_frames"]-p0["any_flag_frames"],"flicker_delta":t1["n_flicker_frames"]-t0["n_flicker_frames"],"boundary_delta":t1["n_boundary_pairs"]-t0["n_boundary_pairs"]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,required=True); ap.add_argument("--out",type=Path,required=True); args=ap.parse_args()
    receipt=load(args.root/"analysis/lipsync_receipt.json"); missing=required(args.root,receipt)
    if missing: raise SystemExit(json.dumps({"missing_inputs":missing},ensure_ascii=False,indent=2))
    people={}; all_side=[]; posthoc=[]
    for sample,row in receipt["people"].items():
        status=g4_person_status(row["windows"]); detail=[]; side=[]
        clean=row["clean_lipsync"]
        for i,(off,conf) in enumerate(zip(clean["seg_offsets"],clean["seg_conf"])):
            if conf is not None and conf>=3 and abs(off)>=10: posthoc.append({"sample":sample,"segment":i,"offset":off,"confidence":conf})
        for w in ("W1","W2"):
            for k,entry in row["windows"][w].items():
                kk=int(k); segs=list(WINDOW_SEGMENTS[w]); confidences=[clean["seg_conf"][i] for i in segs]
                detail.append({"window":w,"k":kk,"inside_deltas_25fps":entry["inside_deltas_25fps"],"clean_confidence":confidences,"median":entry["inside_median_25fps"],"expected":entry["expected_delta_25fps"],"error":None if entry["inside_median_25fps"] is None else entry["inside_median_25fps"]-entry["expected_delta_25fps"],"g4":entry["G4_pass"],"outside_n":entry["outside_n"],"outside_fraction_abs_le_1":None if not entry["outside_n"] else sum(abs(x)<=1 for x in entry["outside_deltas_25fps"])/entry["outside_n"],"g5":entry["G5_pass"]})
                side.append({"window":w,"k":kk,**task_side(args.root,sample,w,kk)})
        all_side.extend(side)
        human=load(Path("/data/lanyi/outputs/cfpr-v2/evaluator_unit_tests_20261108_v2/clean")/sample/"evaluators/lipsync/gen_lipsync.json")
        people[sample]={"g4_registered_status":status,"g5_all_conditions":all(x["g5"] is True for x in detail),"g6":row["G6"],"low_confidence_unit_test":row["low_confidence_in_unit_test"],"details":detail,"side_effects":side,"generated_clean":{"offset_median":med([x for x in clean["seg_offsets"] if x is not None]),"confidence_median":med([x for x in clean["seg_conf"] if x is not None])},"human_clean":{"offset_median":med([x for x in human["seg_offsets"] if x is not None]),"confidence_median":med([x for x in human["seg_conf"] if x is not None] )}}
    others=[people[x]["g4_registered_status"] for x in SAMPLES if x!="CRB100_001"]
    impact=load(args.root/"analysis/impact_and_summary.json")["impact"]
    out={"source":"read_only_run_20261109","g4_registered":{"criterion":"|k| in {4,8}; 2 windows x 4 k per person","people":{s:people[s]["g4_registered_status"] for s in SAMPLES},"others_pass":sum(x=="pass" for x in others),"others_determinable":sum(x!="undetermined" for x in others),"enough_determinable":sum(x!="undetermined" for x in others)>=5},"g4_appendix_report_only":{"criterion":"all k including +/-2","pass":81,"fail":3,"undetermined":12},"people":people,"posthoc_report_only_clean_boundary_offsets":posthoc,"impact_report_only":impact,"side_effects_report_only":{"all_condition_deltas":all_side}}
    args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(out,ensure_ascii=False,indent=2))
    print(json.dumps({"out":str(args.out),"g4":out["g4_registered"]},ensure_ascii=False))
if __name__=="__main__": main()
