"""EchoMimicV2-ACC 生成核心（cfpr-v2 自己的薄封装，不改官方代码）。

去噪循环逐行对照官方 src/pipelines/pipeline_echomimicv2_acc.py:330-521，只用官方推理参数
（768²、6 步 DDIM、cfg 1.0、context 12 / overlap 3、24fps），并留三个干预钩子：
  init_noise_edit(latents)       初始噪声生成后
  bank_edit(writer_modules)      参考 token 写入后、交给去噪网络前
  step_edit(t_i, latents)        每个调度器步之后
  ctx_edit(t_i, frames, reader)  每个 12 帧窗口去噪之前（frames = 本窗实际帧号；可在这里按窗口换读取端的参考 token）
不传钩子时与官方逐字节相同（回归：旧输入 CRB100_001 前 120 帧 raw sha256 前缀 9276500e4a9dc5cc）。
探针里试过的画质开关（分辨率、步数、CFG 等）都证明无益，这里不保留，见 docs/status/acc_findings.md。
"""
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ACC = "/data/lanyi/third_party/echomimic_v2/38c86809efa041884c774ee31d984a9577c0e0aa"
WTS = "/data/lanyi/models/external-pretrained/generators/echomimicv2"
sys.path.insert(0, ACC)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from src.models.unet_2d_condition import UNet2DConditionModel  # noqa: E402
from src.models.unet_3d_emo import EMOUNet3DConditionModel  # noqa: E402
from src.models.whisper.audio2feature import load_audio_model  # noqa: E402
from src.models.pose_encoder import PoseEncoder  # noqa: E402
from src.models.mutual_self_attention import ReferenceAttentionControl, torch_dfs  # noqa: E402
from src.models.attention import BasicTransformerBlock  # noqa: E402
from src.pipelines.pipeline_echomimicv2_acc import EchoMimicV2Pipeline  # noqa: E402
from src.pipelines.context import get_context_scheduler  # noqa: E402
from src.pipelines.step_func import origin_by_velocity_and_sample, psuedo_velocity_wrt_noisy_and_timestep  # noqa: E402
from src.utils.dwpose_util import draw_pose_select_v2  # noqa: E402

W = H = 768
FPS, STEPS, SEED, CTX, OVL = 24, 6, 3407, 12, 3


def load_pipe(dtype=torch.float16):
    icfg = OmegaConf.load(f"{ACC}/configs/inference/inference_v2.yaml")
    base = f"{WTS}/sd-image-variations-diffusers"
    vae = AutoencoderKL.from_pretrained(f"{base}/vae").to("cuda", dtype=dtype)   # 权重与官方 sd-vae-ft-mse 逐张量相同
    ref = UNet2DConditionModel.from_pretrained(base, subfolder="unet").to("cuda", dtype=dtype)
    ref.load_state_dict(torch.load(f"{WTS}/reference_unet.pth", map_location="cpu"))
    den = EMOUNet3DConditionModel.from_pretrained_2d(base, f"{WTS}/motion_module_acc.pth", subfolder="unet",
                                                     unet_additional_kwargs=icfg.unet_additional_kwargs).to("cuda", dtype=dtype)
    den.load_state_dict(torch.load(f"{WTS}/denoising_unet_acc.pth", map_location="cpu"), strict=False)
    pose = PoseEncoder(320, conditioning_channels=3, block_out_channels=(16, 32, 96, 256)).to("cuda", dtype=dtype)
    pose.load_state_dict(torch.load(f"{WTS}/pose_encoder.pth"))
    audio = load_audio_model(model_path=f"{WTS}/audio_processor/tiny.pt", device="cuda")
    sched = DDIMScheduler(**OmegaConf.to_container(icfg.noise_scheduler_kwargs))
    return EchoMimicV2Pipeline(vae=vae, reference_unet=ref, denoising_unet=den, audio_guider=audio,
                               pose_encoder=pose, scheduler=sched).to("cuda", dtype=dtype)


def pose_frame(d):
    """同 infer_acc.py：官方 draw_pose_select_v2 只画手。"""
    imh, imw, rb, re, cb, ce = d["draw_pose_params"]
    im = np.transpose(np.array(draw_pose_select_v2(d, imh, imw, ref_w=800)), (1, 2, 0))
    t = np.zeros((W, H, 3), dtype=np.uint8)
    t[rb:re, cb:ce, :] = im
    return torch.Tensor(np.array(Image.fromarray(t).convert("RGB"))).permute(2, 0, 1) / 255.0


def load_inputs(input_dir, L):
    """input_dir 下需有 reference.png、audio.wav、pose/<i>.npy（src/prepare_aligned.py 的产物）。"""
    d = Path(input_dir)
    ref = Image.open(d / "reference.png").convert("RGB")
    frames = [pose_frame(np.load(d / "pose" / f"{i}.npy", allow_pickle=True).tolist()) for i in range(L)]
    return ref, str(d / "audio.wav"), torch.stack(frames, dim=1).unsqueeze(0).to("cuda", torch.float16)


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()


def writer_modules(pipe):
    # 与 mutual_self_attention.py:298-308 的写入端排序一致
    return sorted([m for m in torch_dfs(pipe.reference_unet) if isinstance(m, BasicTransformerBlock)],
                  key=lambda x: -x.norm1.normalized_shape[0])


@torch.no_grad()
def generate(pipe, ref_img, audio_path, poses, L, seed=SEED, init_noise_edit=None, bank_edit=None, step_edit=None, ctx_edit=None,
             audio_edit=None):
    dev, dt = torch.device("cuda"), pipe.vae.dtype
    gen = torch.manual_seed(seed)                                    # infer_acc.py:167
    pipe.scheduler.set_timesteps(STEPS, device=dev)
    ts = pipe.scheduler.timesteps
    writer = ReferenceAttentionControl(pipe.reference_unet, do_classifier_free_guidance=False, mode="write",
                                       batch_size=1, fusion_blocks="full")
    reader = ReferenceAttentionControl(pipe.denoising_unet, do_classifier_free_guidance=False, mode="read",
                                       batch_size=1, fusion_blocks="full")
    wf = pipe.audio_guider.audio2feat(audio_path)
    audio = torch.Tensor(pipe.audio_guider.feature2chunks(feature_array=wf, fps=FPS)).to(dtype=dt, device=dev).unsqueeze(0)
    L = min(L, audio.shape[1])
    if audio_edit is not None:
        edited_audio = audio_edit(audio)
        if edited_audio.shape != audio.shape:
            raise ValueError(f"audio_edit must preserve shape {tuple(audio.shape)}, got {tuple(edited_audio.shape)}")
        audio = edited_audio
    lat = pipe.prepare_latents_smooth(1, pipe.denoising_unet.in_channels, W, H, L, audio.dtype, dev, gen, CTX)
    if init_noise_edit:
        lat = init_noise_edit(lat)
    init_sha = sha(lat)
    pose_fea = pipe.pose_encoder(poses[:, :, :L])
    extra = pipe.prepare_extra_step_kwargs(gen, 0.0)
    rt = pipe.ref_image_processor.preprocess(ref_img, height=H, width=W).to(dtype=dt, device=dev)
    ref_lat = pipe.vae.encode(rt).latent_dist.mean * 0.18215
    queue = list(get_context_scheduler("uniform")(0, STEPS, lat.shape[2], CTX, 1, OVL))
    step_sha, step_lat = [], []
    for t_i, t in enumerate(ts):
        npred = torch.zeros_like(lat)
        cnt = torch.zeros((1, 1, lat.shape[2], 1, 1), device=dev, dtype=lat.dtype)
        if t_i == 0:
            pipe.reference_unet(ref_lat, torch.zeros_like(t), encoder_hidden_states=None, return_dict=False)
            if bank_edit:
                bank_edit(writer_modules(pipe))
            reader.update(writer, do_classifier_free_guidance=False)
        for ctx in queue:                                            # context_batch_size = 1
            c = [(x + t_i * 3) % L for x in ctx]
            if ctx_edit:
                ctx_edit(t_i, c, reader)
            inp = pipe.scheduler.scale_model_input(lat[:, :, c], t)
            pred = pipe.denoising_unet(inp, t, encoder_hidden_states=None, audio_cond_fea=audio[:, c],
                                       face_musk_fea=pose_fea[:, :, c], return_dict=False)[0]
            ac = pipe.scheduler.alphas_cumprod.to(dev)
            x0 = origin_by_velocity_and_sample(pred, inp, ac, t)
            pred = psuedo_velocity_wrt_noisy_and_timestep(inp, x0, ac, t, torch.ones_like(t) * (-1))
            npred[:, :, c] += pred
            cnt[:, :, c] += 1
        lat = pipe.scheduler.step(npred / cnt, t, lat, **extra).prev_sample
        if step_edit:
            lat = step_edit(t_i, lat)
        step_sha.append(sha(lat))
        step_lat.append(lat.detach().clone())
    reader.clear()
    writer.clear()
    vid = torch.from_numpy(pipe.decode_latents(lat))                 # (1, 3, L, H, W) float [0,1]
    return vid, {"L": L, "seed": seed, "init_noise_sha256": init_sha, "step_latent_sha256": step_sha}, step_lat


def save(out_dir, vid, info, audio_path, final_latents=None):
    """逐帧无损 PNG + 每帧 sha256 + frames_uint8.npy + 预览 mp4 + summary.json。"""
    d = Path(out_dir)
    (d / "frames").mkdir(parents=True, exist_ok=True)
    fr = (vid[0].permute(1, 2, 3, 0).numpy() * 255).round().astype(np.uint8)
    def _png(i):                                                     # 8 线程并行写 PNG（内容不变；实测 240 帧 63 s → 9 s）
        Image.fromarray(fr[i]).save(d / "frames" / f"{i:04d}.png")
        return hashlib.sha256(fr[i].tobytes()).hexdigest()
    with ThreadPoolExecutor(8) as ex:
        shas = list(ex.map(_png, range(len(fr))))
    np.save(d / "frames_uint8.npy", fr)
    if final_latents is not None:
        np.save(d / "final_latents_fp16.npy", final_latents.detach().cpu().numpy())
    ff = Path(sys.executable).parent / "ffmpeg"
    subprocess.run([str(ff), "-hide_banner", "-loglevel", "error", "-y", "-framerate", str(FPS), "-i", str(d / "frames/%04d.png"),
                    "-i", audio_path, "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                    str(d / "preview.mp4")], check=True)
    info.update({"raw_output_sha256": sha(vid), "frame_sha256": shas, "audio": audio_path,
                 "torch_max_allocated_mib": torch.cuda.max_memory_allocated() // 2**20})
    (d / "summary.json").write_text(json.dumps(info, indent=1, ensure_ascii=False))
    return info
