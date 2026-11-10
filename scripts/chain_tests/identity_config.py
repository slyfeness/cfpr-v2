"""Frozen task registry for the identity-drift chain experiment."""
SAMPLES=("CRB100_001","CRB100_002","CRB100_025","CRB100_050","CRB100_056","CRB100_075","CRB100_087","CRB100_090")
DONOR={sample:SAMPLES[(i+1)%len(SAMPLES)] for i,sample in enumerate(SAMPLES)}
WINDOWS={"W1":(96,144),"W2":(168,216)}
ALPHAS=(.05,.10,.20,.40)

def build_tasks(seed=3407, frames=240):
    tasks=[]
    for sample in SAMPLES:
        donor=DONOR[sample]
        tasks += [
            {"sample":sample,"frames":frames,"seed":seed,"kind":"clean_a","actions":[],"out_id":f"{sample}/clean_a"},
            {"sample":sample,"frames":frames,"seed":seed,"kind":"clean_b","actions":[],"out_id":f"{sample}/clean_b"},
            {"sample":sample,"frames":frames,"seed":seed,"kind":"zero_action","actions":[{"type":"face_bank_mix","window":list(WINDOWS["W1"]),"alpha":0.0,"donor":donor}],"out_id":f"{sample}/zero_action"},
        ]
        for w,window in WINDOWS.items():
            for alpha in ALPHAS:
                tasks.append({"sample":sample,"frames":frames,"seed":seed,"kind":f"{w}_a_{alpha:.2f}","actions":[{"type":"face_bank_mix","window":list(window),"alpha":alpha,"donor":donor}],"out_id":f"{sample}/{w}/a_{alpha:.2f}"})
    return tasks
