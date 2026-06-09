# ---------------------------------------------------------------------------
# config.py  –  all tunable constants for flux-stream-webcam (FLUX.2-Klein-4B)
# ---------------------------------------------------------------------------
#
# This is the pure-Python / diffusers replacement for the old ComfyUI workflow
# (webcam-voice-i2i).  Model + sampler settings mirror the proven realtime
# ComfyUI graph: FLUX.2-Klein-4B, 2 steps, euler, cfg/guidance = 1.0, 512x512,
# fp8 transformer.  Klein is an *edit/reference* model — the webcam frame is
# encoded and injected as a reference latent, so there is no img2img "strength"
# knob; STEPS is the main speed/quality dial.
# ---------------------------------------------------------------------------

# Model — FLUX.2-Klein-4B (Apache 2.0, distilled, 4-step capable).
# Loaded in diffusers format from HF cache; bundles its own Qwen3-4B text
# encoder (~7.5 GB) + VAE.  Whole pipeline is ~15 GB on the GPU.
MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"

# Inference device
DEVICE = "cuda"

# Output / capture resolution (multiples of 16). Lower = much faster (fewer
# tokens). Measured on the 5090 @ 1-step / 2-step:
#   512 -> 8.4 / 5.5 FPS   |   384 -> 14 / 9 FPS   |   320 -> 18 / 12 FPS
# The output is upscaled to the display window.
# NOTE: changing this requires a restart (triggers a torch.compile recompile).
WIDTH  = 512
HEIGHT = 512

# Diffusion settings
STEPS    = 2      # euler steps. 1 = fastest, 2 = good default, 4 = max quality. Live: +/-
GUIDANCE = 1.0    # cfg. Klein distilled wants 1.0 — no true-CFG, no negatives

# Structure-strength dial (img2img). How far the webcam frame is noised before
# denoising: low = webcam barely changed, high = wild reimagining by the prompt.
# Live-adjustable with [ and ] keys. Keep >= 0.4 so at least one step actually runs.
STRENGTH = 0.9

# Temporal feedback (0 = off). Each frame the img2img input is a blend of the
# live webcam and the PREVIOUS styled output: input = (1-FEEDBACK)*webcam +
# FEEDBACK*prev_output.  Higher = more trails / morphing / evolving visuals even
# when you hold still.  Live-adjustable with , and . keys.
FEEDBACK = 0.0

# Fixed sampling seed — reusing the same noise every frame greatly reduces
# frame-to-frame flicker, which matters far more for live video than variety.
SEED = 859840788834783

# Speed optimisations
USE_FP8     = True    # fp8 (PerTensor) dynamic quant of the transformer (Blackwell native)
USE_COMPILE = True    # torch.compile(max-autotune) the transformer — big win, slow first warmup

# Whisper model size for live transcription
WHISPER_MODEL = "base.en"   # tiny.en = faster, base.en = more accurate

# Audio input device — None = system default. Device INDICES shift when devices
# connect/disconnect, so prefer a NAME substring (robust): e.g.
#   AUDIO_DEVICE = "Microphone"   or   AUDIO_DEVICE = "Webcam"
# Run MIC_CHECK.bat to list devices and confirm levels.
AUDIO_DEVICE = None

# ---- Audio capture / transcription tuning (ported from webcam-voice-i2i) ----
AUDIO_SAMPLE_RATE      = 16_000  # whisper native rate
AUDIO_BUFFER_SECONDS   = 3       # rolling buffer length fed to Whisper
AUDIO_TRANSCRIBE_INTERVAL = 1.0  # seconds between Whisper runs
# Silence gate: measured as the loudest 0.5 s in the window (NOT the whole-window
# average), so short phrases aren't diluted by surrounding silence. Lower if it
# ignores you; raise if background noise keeps triggering. 0 disables the gate.
# Watch the console: it prints the live level so you can pick a good value.
AUDIO_SILENCE_THRESHOLD   = 0.006
AUDIO_MIN_WORDS        = 1       # reject transcripts shorter than this many words
AUDIO_USE_VAD          = True    # faster-whisper Silero VAD (extra speech filter)
AUDIO_DEBUG            = True     # print live mic level + why a phrase was skipped
VOICE_DEBOUNCE         = 1.0     # seconds transcribed text must hold steady before it commits

# ---- LoRA stack -------------------------------------------------------------
# LoRAs are FUSED into the model at launch (before fp8 + compile), so they keep
# full realtime speed. Multiple entries stack/blend by their weight. Changing
# this list requires a restart (re-fuse + recompile).  Must be FLUX.2-Klein-4B
# LoRAs — others won't load.
LORA_DIR = r"D:\AI_software\ComfyUI_windows_portable\ComfyUI\models\loras\flux2_klein"
LORAS = [
    ("OctaneRenderKlein4BF.safetensors", 1.0),
    # ("RebelReal (KLEIN 4b).safetensors", 0.4),   # stack a second one like this
]

# Trigger words auto-appended to every prompt so the LoRA always activates.
# (Octane Render LoRA's trigger is "Octane Render".)  Set "" to disable.
PROMPT_SUFFIX = "Octane Render"

# Short label for the HUD (what LoRA look is active)
LORA_LABEL = "Octane"

# First prompt shown before you speak
INITIAL_PROMPT = "a marble bust"
