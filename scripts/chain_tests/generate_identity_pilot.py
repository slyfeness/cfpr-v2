#!/usr/bin/env python3
"""Direct P0--P7 pilot generator using only ACC's existing ctx_edit hook."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import cv2
import torch
from PIL import Image

from scripts.chain_tests.identity_pilot import WINDOW, make_pilot_ctx_edit, pilot_specs
from src import acc_core
from src.models.mutual_self_attention import ReferenceAttentionControl, torch_dfs


INPUT = Path("/data/lanyi/outputs/cfpr-v2/prepared_inputs_aligned_v1")
STEP3 = Path("/data/lanyi/outputs/cfpr-v2/chain_identity_drift/run_20261109")
DEFAULT_ROOT = Path("/data/lanyi/outputs/cfpr-v2/identity_injection_pilot/run_20261110")
PAIRS = {"CRB100_001": "CRB100_002", "CRB100_025": "CRB100_050"}
SEED, FRAMES = 3407, 240


def reader_modules(pipe):
    return sorted([module for module in torch_dfs(pipe.denoising_unet) if type(module).__name__ == "TemporalBasicTransformerBlock"], key=lambda x: -x.norm1.normalized_shape[0])


@torch.no_grad()
def bank_of(pipe, image):
    writer = ReferenceAttentionControl(pipe.reference_unet, do_classifier_free_guidance=False, mode="write", batch_size=1, fusion_blocks="full")
    latent = pipe.ref_image_processor.preprocess(image, height=acc_core.H, width=acc_core.W).to(dtype=pipe.vae.dtype, device="cuda")
    pipe.reference_unet(pipe.vae.encode(latent).latent_dist.mean * .18215, torch.zeros(1, device="cuda", dtype=torch.long), encoder_hidden_states=None, return_dict=False)
    bank = [module.bank[0].clone() for module in acc_core.writer_modules(pipe)]
    writer.clear()
    return bank


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_p7_reference(own_path: Path, aligned_donor_path: Path, box, out_path: Path):
    own, donor = cv2.imread(str(own_path)), cv2.imread(str(aligned_donor_path))
    if own is None or donor is None:
        raise FileNotFoundError(f"P7 reference source missing: {own_path} / {aligned_donor_path}")
    height, width = own.shape[:2]
    x0, y0, x1, y1 = [int(round(value * size)) for value, size in zip(box, (width, height, width, height))]
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width - 1, x1), min(height - 1, y1)
    mask = cv2.ellipse(__import__("numpy").zeros((height, width), dtype="uint8"), ((x0 + x1) // 2, (y0 + y1) // 2), (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), 0, 0, 360, 255, -1)
    clone = cv2.seamlessClone(donor, own, mask, ((x0 + x1) // 2, (y0 + y1) // 2), cv2.NORMAL_CLONE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), clone):
        raise RuntimeError(f"failed writing {out_path}")
    return {"own_reference": str(own_path), "aligned_donor_reference": str(aligned_donor_path), "face_box": list(box), "method": "cv2.seamlessClone NORMAL_CLONE ellipse", "sha256": sha256(out_path)}


def link_clean(root: Path, sample: str):
    source = STEP3 / sample / "clean_a"
    expected = json.loads((source / "summary.json").read_text())["raw_output_sha256"]
    actual = json.loads((source / "summary.json").read_text())["raw_output_sha256"]
    if actual != expected:
        raise RuntimeError(f"clean hash mismatch for {sample}")
    target = root / sample / "clean_a"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            raise FileExistsError(target)
    else:
        target.symlink_to(source, target_is_directory=True)
    return {"source": str(source), "raw_output_sha256": actual, "verified_against_step3_record": True}


def run_variant(pipe, root, sample, donor, alignment, spec, p7_info):
    output = root / sample / spec["id"]
    if output.exists():
        raise FileExistsError(output)
    own_ref = INPUT / sample / "reference.png"
    aligned_path = STEP3 / "alignment" / f"{sample}__donor_{donor}_aligned.png"
    reference_path = root / sample / "P7_reference.png"
    if spec["id"] == "P7":
        make_p7_reference(own_ref, aligned_path, alignment[sample]["own_face_box"], reference_path)
        full_image = Image.open(reference_path).convert("RGB")
    else:
        full_image = Image.open(INPUT / donor / "reference.png").convert("RGB")
    ref, audio, poses = acc_core.load_inputs(INPUT / sample, FRAMES)
    aligned_bank = bank_of(pipe, Image.open(aligned_path).convert("RGB"))
    full_bank = bank_of(pipe, full_image)
    action = dict(spec, donor=donor)
    edit, state = make_pilot_ctx_edit(reader_modules(pipe), aligned_bank, full_bank, alignment[sample]["own_face_box"], action)
    torch.cuda.reset_peak_memory_stats()
    video, info, steps = acc_core.generate(pipe, ref, audio, poses, FRAMES, seed=SEED, ctx_edit=edit)
    info["identity_injection_pilot"] = {"sample": sample, "donor": donor, "variant": spec, "clean_reused": str(root / sample / "clean_a")}
    acc_core.save(output, video, info, audio, final_latents=steps[-1])
    metadata = {"sample": sample, "donor": donor, "variant": spec, "window": list(WINDOW), "contexts": state["contexts"], "layer_token_counts": state["layer_token_counts"], "face_token_counts": [int(mask.sum().item()) for mask in state["face_masks"]]}
    if spec["id"] == "P7":
        metadata["p7_reference"] = make_p7_reference(own_ref, aligned_path, alignment[sample]["own_face_box"], reference_path)
    (output / "edit_meta.json").write_text(json.dumps(metadata, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--variants", nargs="*")
    args = parser.parse_args()
    selected = [sample for sample in PAIRS if not args.samples or sample in args.samples]
    variants = [spec for spec in pilot_specs() if not args.variants or spec["id"] in args.variants]
    args.out_root.mkdir(parents=True, exist_ok=True)
    alignment = json.loads((STEP3 / "alignment" / "alignment_manifest.json").read_text())
    clean = {sample: link_clean(args.out_root, sample) for sample in selected}
    (args.out_root / "clean_reuse_audit.json").write_text(json.dumps(clean, indent=2))
    pipe = acc_core.load_pipe()
    try:
        for sample in selected:
            for spec in variants:
                run_variant(pipe, args.out_root, sample, PAIRS[sample], alignment, spec, None)
                print(json.dumps({"sample": sample, "variant": spec["id"], "status": "complete"}), flush=True)
    finally:
        del pipe
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
