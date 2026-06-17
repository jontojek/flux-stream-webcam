# ---------------------------------------------------------------------------
# infer.py  –  FLUX.2-Klein-4B img2img inference (fp8 + torch.compile)
# ---------------------------------------------------------------------------
#
# We use Flux2KleinInpaintPipeline as a plain img2img pipeline (full-white mask,
# no image_reference).  The webcam frame becomes the *initial* latent; `strength`
# controls how far it's noised before denoising — i.e. a live structure dial:
#   low  strength → webcam barely changed (structure preserved)
#   high strength → wild reimagining driven by the spoken prompt
# Sampling uses a fixed seed for a stable, low-flicker feed.  Speed is governed
# by STEPS and resolution (see config).
# ---------------------------------------------------------------------------

import os

import numpy as np
import torch
from PIL import Image
from diffusers import Flux2KleinInpaintPipeline

import config

COMPILED = config.USE_COMPILE  # for status reporting

# ---------------------------------------------------------------------------
# Schedule mu override (FAL's realtime Klein demo calls this `schedule_mu`:
# https://fal.ai/models/fal-ai/flux-2/klein/realtime). Klein's pipeline always
# derives the flow-matching scheduler's "mu" time-shift from resolution + step
# count via compute_empirical_mu() -- there's no kwarg on __call__ to override
# it. compute_empirical_mu is referenced as a bare module-level name inside
# the pipeline's __call__, so monkeypatching that name on the live module
# object intercepts every call without forking/subclassing the pipeline.
# config.SCHEDULE_MU = None keeps the original (auto) behaviour untouched.
# ---------------------------------------------------------------------------
try:
    import diffusers.pipelines.flux2.pipeline_flux2_klein_inpaint as _flux2_klein_mod

    _original_compute_empirical_mu = _flux2_klein_mod.compute_empirical_mu

    def _compute_empirical_mu_override(image_seq_len: int, num_steps: int) -> float:
        if config.SCHEDULE_MU is not None:
            return float(config.SCHEDULE_MU)
        return _original_compute_empirical_mu(image_seq_len, num_steps)

    _flux2_klein_mod.compute_empirical_mu = _compute_empirical_mu_override
    print("[infer] schedule_mu override installed (config.SCHEDULE_MU; live keys m/n/r).")
except Exception as e:
    print(f"[infer] schedule_mu override NOT installed ({e}) -- "
          f"SCHEDULE_MU will be ignored; auto behaviour unaffected.")

# ---------------------------------------------------------------------------
# Prompt embedding cache — skip the ~40 ms Qwen3 text encode when the prompt is
# unchanged (almost every frame).  Cached embeds are passed straight back in.
# ---------------------------------------------------------------------------
_cache_text: str | None = None
_cache_embeds: torch.Tensor | None = None
_white_mask: Image.Image | None = None


def _get_embeds(pipe: Flux2KleinInpaintPipeline, prompt: str) -> torch.Tensor:
    global _cache_text, _cache_embeds
    if prompt == _cache_text and _cache_embeds is not None:
        return _cache_embeds
    print(f"[infer] Encoding prompt: {prompt!r}")
    with torch.inference_mode():
        prompt_embeds, _text_ids = pipe.encode_prompt(
            prompt=prompt, device=config.DEVICE, num_images_per_prompt=1,
        )
    _cache_text = prompt
    _cache_embeds = prompt_embeds
    return _cache_embeds


def build_pipeline() -> Flux2KleinInpaintPipeline:
    """Load FLUX.2-Klein-4B img2img pipeline, apply fp8 quant + torch.compile."""
    print("[infer] Loading FLUX.2-Klein-4B (img2img) pipeline ...")
    pipe = Flux2KleinInpaintPipeline.from_pretrained(
        config.MODEL_ID, torch_dtype=torch.bfloat16,
    ).to(config.DEVICE)
    pipe.set_progress_bar_config(disable=True)

    # --- LoRA: load + blend by weight, then FUSE before fp8/compile ----------
    # Fusing bakes the LoRAs into the bf16 weights so they survive quantisation
    # and add zero runtime cost. Order matters: fuse must happen before fp8.
    loras = [l for l in getattr(config, "LORAS", []) if l]
    if loras:
        names, weights = [], []
        for i, (fn, w) in enumerate(loras):
            path = fn if os.path.isabs(fn) else os.path.join(config.LORA_DIR, fn)
            name = f"lora{i}"
            print(f"[infer] Loading LoRA {fn!r} (weight {w}) ...")
            pipe.load_lora_weights(path, adapter_name=name)
            names.append(name); weights.append(w)
        pipe.set_adapters(names, adapter_weights=weights)
        pipe.fuse_lora()
        pipe.unload_lora_weights()
        print(f"[infer] Fused {len(names)} LoRA(s): "
              f"{', '.join(f'{fn} @ {w}' for (fn, w) in loras)}")

    if config.USE_FP8:
        from torchao.quantization import (
            quantize_, Float8DynamicActivationFloat8WeightConfig, PerTensor,
        )
        print("[infer] Quantising transformer -> fp8 dynamic (PerTensor) ...")
        quantize_(pipe.transformer,
                  Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()))

    if config.USE_COMPILE:
        print("[infer] torch.compile(transformer + vae decode) — first warmup is slow ...")
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune-no-cudagraphs", fullgraph=True)
        try:
            pipe.vae.decode = torch.compile(
                pipe.vae.decode, mode="max-autotune-no-cudagraphs")
        except Exception as e:
            print(f"[infer] vae compile skipped: {e}")

    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    used = torch.cuda.memory_allocated(0) / 1024**3
    print(f"[infer] GPU: {torch.cuda.get_device_name(0)}  VRAM: {used:.1f} / {vram:.1f} GB")
    print(f"[infer] fp8={config.USE_FP8} compile={config.USE_COMPILE} "
          f"res={config.WIDTH}x{config.HEIGHT} steps={config.STEPS} strength={config.STRENGTH}")
    return pipe


def warmup(pipe: Flux2KleinInpaintPipeline, prompt: str, passes: int | None = None) -> None:
    """Pre-encode the prompt and warm/compile GPU kernels."""
    if passes is None:
        passes = 4 if config.USE_COMPILE else 1
    print(f"[infer] Warming up ({passes} pass{'es' if passes != 1 else ''}) ...")
    dummy = np.zeros((config.HEIGHT, config.WIDTH, 3), dtype=np.uint8)
    for i in range(passes):
        infer(pipe, dummy, prompt)
        torch.cuda.synchronize()
        if config.USE_COMPILE:
            print(f"[infer]   warmup pass {i + 1}/{passes} done")
    print("[infer] Warmup done — live loop starting.")


def _get_mask() -> Image.Image:
    global _white_mask
    if _white_mask is None or _white_mask.size != (config.WIDTH, config.HEIGHT):
        _white_mask = Image.new("L", (config.WIDTH, config.HEIGHT), 255)
    return _white_mask


def _bgr_to_pil(frame: np.ndarray) -> Image.Image:
    return Image.fromarray(frame[:, :, ::-1])


def _pil_to_bgr(image: Image.Image) -> np.ndarray:
    return np.array(image)[:, :, ::-1]


def infer(pipe: Flux2KleinInpaintPipeline, frame_bgr: np.ndarray, prompt: str) -> np.ndarray:
    """One img2img pass. frame_bgr: H×W×3 uint8 BGR. Returns H×W×3 uint8 BGR."""
    suffix = getattr(config, "PROMPT_SUFFIX", "")
    full_prompt = f"{prompt}, {suffix}" if suffix else prompt
    prompt_embeds = _get_embeds(pipe, full_prompt)

    pil_in = _bgr_to_pil(frame_bgr).resize(
        (config.WIDTH, config.HEIGHT), Image.LANCZOS)

    # Fixed seed every frame → identical noise → stable, low-flicker feed.
    generator = torch.Generator(device=config.DEVICE).manual_seed(config.SEED)

    with torch.inference_mode():
        result = pipe(
            image=pil_in,
            mask_image=_get_mask(),
            image_reference=None,
            prompt_embeds=prompt_embeds,
            num_inference_steps=config.STEPS,
            guidance_scale=config.GUIDANCE,
            strength=config.STRENGTH,
            height=config.HEIGHT,
            width=config.WIDTH,
            generator=generator,
        ).images[0]

    return _pil_to_bgr(result)
