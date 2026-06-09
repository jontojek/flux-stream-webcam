# 🎨 flux-stream-webcam

> **Speak a style out loud and watch your webcam transform into it — live.**

A real-time, voice-driven video style-transfer toy for Windows + NVIDIA GPUs. Talk
into your mic ("a bronze statue", "cyberpunk neon city", "carved jade"), and your
webcam feed is continuously re-rendered in that style by the **FLUX.2-Klein-4B**
image model. Add a LoRA for material looks, dial how much of your real face shows
through, and turn on feedback for trippy morphing.

It's a pure **Python + 🤗 diffusers** app — no ComfyUI, no nodes, just run it.

<!-- TODO: add a screenshot or GIF here, e.g. ![demo](docs/demo.gif) -->

---

## ✨ What it does

- 🎙️ **Voice control** — your speech is transcribed live (Whisper) and becomes the art prompt.
- 🖼️ **Real-time restyle** — every webcam frame is re-imagined by FLUX.2-Klein-4B (~5–14 FPS depending on settings).
- 🎚️ **Structure dial** — slide from "barely-touched webcam" to "totally reimagined".
- 🌀 **Temporal feedback** — feed the last frame back in for trails / morphing motion.
- 🧱 **LoRA support** — drop in style/material LoRAs (e.g. an *Octane Render* look) for transformations prompts alone can't do.
- ⚡ **Optimized** — fp8 quantization + `torch.compile` to keep it interactive.

---

## 🖥️ What you need

| | |
|---|---|
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA, **16 GB VRAM or more** (developed on an RTX 5090, 32 GB) |
| **Disk** | ~20 GB free (the model is ~15 GB) |
| **Python** | A working Python 3.11+ install (this project uses Miniconda) |
| **Webcam + mic** | Any that Windows recognizes |

> 💡 **Beginner note:** this needs a fairly powerful *NVIDIA* graphics card. It will
> not run on a laptop with only integrated graphics, or on an AMD/Mac GPU.

---

## 🚀 Quick start

### 1. Install the Python packages (one time)

Double-click **`INSTALL.bat`**. It installs PyTorch and everything else into the
Python it points at. This takes a while (a few GB of downloads) — be patient.

> ⚠️ **The most common beginner mistake:** running the app with the *wrong* Python.
> This machine has several Python installs, and only one has the packages. The
> `.bat` files always use the correct one for you. If you ever run a command by
> hand, use the **full path**, not a bare `python`:
> ```
> C:\Users\jonto\miniconda3\python.exe main.py
> ```

### 2. Run it

Double-click **`START.bat`**.

- **First launch** does two slow-but-one-time things:
  1. **Downloads the model** (~15 GB) from Hugging Face into `models\hf_cache\`. No
     account or login needed — the model is free/open.
  2. **Compiles GPU kernels** (1–3 minutes). You'll see a "Warming up…" window.
- **Every launch after** is fast (both are cached).

When the window shows your styled webcam, **start talking**. Say a style and watch
it change.

Press **`Q`** (or close the window) to quit.

---

## 🎮 Controls

Everything is shown live on the top bar of the window, so you don't have to memorize it.

| Input | What it does |
|---|---|
| 🎙️ **Speak** | Sets the art style (e.g. "a watercolor sunset") |
| **`+`** / **`-`** | More / fewer diffusion **steps** — higher = nicer but slower |
| **`]`** / **`[`** | **Strength** up/down — how much it restyles vs. keeps your real webcam |
| **`.`** / **`,`** | **Feedback** up/down — trails & morphing (0 = off) |
| **`Q`** / **`Esc`** | Quit |

The on-screen status line reads: `fps · steps · str(ength) · fb(feedback) · lora`.

---

## 🎛️ Tuning it — `config.py`

All the knobs live in **`config.py`** (a plain text file you can edit in Notepad).
The most useful ones:

| Setting | What it means | Tips |
|---|---|---|
| `WIDTH` / `HEIGHT` | Output resolution | **Biggest speed lever.** `512` = best look (~5 FPS), `384` ≈ 9 FPS, `320` ≈ 12–18 FPS. Restart after changing. |
| `STEPS` | Diffusion steps per frame | `1` fastest, `2` good default, `4` max quality. Also live via `+`/`-`. |
| `STRENGTH` | How far it reimagines your webcam (0–1) | `~0.5` = subtle, `~0.9` = full restyle. Also live via `[`/`]`. |
| `FEEDBACK` | Blend in the previous frame (0–1) | `0` off; `0.3–0.6` nice morphing; `0.8+` psychedelic. Also live via `,`/`.`. |
| `WHISPER_MODEL` | Speech recognition model | `"base.en"` (default) or `"tiny.en"` for speed. |
| `AUDIO_DEVICE` | Which microphone to use | `None` = system default, or a name like `"Microphone"` (see below). |
| `LORAS` | List of LoRAs to load | See the LoRA section. |

> After editing `config.py`, just relaunch `START.bat`. Resolution/LoRA changes
> trigger a one-time recompile (slow first run again).

---

## 🧱 Material transformations & LoRAs

A **LoRA** is a small add-on file that teaches the model a specific look or style.
This project ships configured for an **Octane Render** LoRA (a hyper-real 3D-render
aesthetic), which is fantastic for **material transformations** — speak a material
and your face/scene becomes it:

> "a bronze warrior" · "carved jade" · "molten glass" · "polished marble" ·
> "liquid chrome" · "weathered copper" · "obsidian" · "crystal"

### Adding your own LoRAs

1. **Get a LoRA trained for FLUX.2-Klein-4B.** ⚠️ It *must* be for this exact model
   — LoRAs for FLUX.1, SDXL, Qwen, etc. will **not** load.
2. **Put the `.safetensors` file** in your LoRA folder (path set by `LORA_DIR` in
   `config.py`).
3. **Add a line to `LORAS`** in `config.py` with a blend weight (0–1):
   ```python
   LORAS = [
       ("OctaneRenderKlein4BF.safetensors", 1.0),
       ("YourNewLoRA.safetensors",          0.6),   # stacks/blends with the first
   ]
   ```
4. **If the LoRA has a "trigger word"**, add it so it always activates, and update
   the on-screen label:
   ```python
   PROMPT_SUFFIX = "Octane Render, <your trigger>"
   LORA_LABEL    = "Octane+New"
   ```
5. **Restart** `START.bat`. The console prints `Fused N LoRA(s): …` to confirm.

> 💡 Tune one LoRA at a time. Two strong LoRAs can fight — start a second one
> around `0.4–0.6`. Comment a line out with `#` to disable it without deleting it.

---

## 🎤 Microphone troubleshooting

If it seems to **ignore you when you talk**, run **`MIC_CHECK.bat`**. It lists every
audio device and shows a live level meter — talk and watch the bar move.

- **Bar doesn't move when you talk** → wrong mic, or the mic is disabled. Check
  *Windows Settings → System → Sound → Input* and *Privacy → Microphone*. Then find
  your mic in the `MIC_CHECK.bat` list and set it **by name** in `config.py`:
  ```python
  AUDIO_DEVICE = "Microphone"   # any part of the device name; survives re-plugging
  ```
- **Bar moves but it still ignores you** → lower `AUDIO_SILENCE_THRESHOLD` in
  `config.py` (try `0.004`). With `AUDIO_DEBUG = True` the console tells you exactly
  why a phrase was skipped (too quiet / filtered / too short).

> Avoid using a numeric device index — Windows renumbers devices when things plug/
> unplug, so a name is more reliable.

---

## 🛠️ Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'numpy'` (or torch, etc.) | You ran the wrong Python. Use the `.bat` files, or the full `miniconda3\python.exe` path. |
| Webcam window is black / "could not open webcam" | Another app (Zoom, Teams, OBS, Chrome) is using the camera. Close it. |
| First run hangs for minutes | Normal — it's downloading the model and/or compiling kernels. Only happens once. |
| Out-of-memory / crash on load | Your GPU has too little VRAM. Lower `WIDTH`/`HEIGHT` to `384` or `320` in `config.py`. |
| Voice not picked up | See the **Microphone troubleshooting** section above. |
| It's too slow | Press `-` (fewer steps), or set a lower resolution in `config.py`. |

---

## 🧠 How it works (the short version)

<details>
<summary>Click to expand the nerdy bits</summary>

- **Model:** [FLUX.2-Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
  by Black Forest Labs — a small (4-billion-parameter), distilled image model that
  works in just 1–2 steps, which is what makes real-time possible. It's Apache-2.0
  licensed and downloads automatically.
- **The loop:** grab a webcam frame → (optionally blend in the last output =
  *feedback*) → run **img2img** (the frame is the starting point; how far it's
  re-noised = *strength*) → the spoken prompt steers the style → show the result.
- **Voice:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs on a
  background thread, gated by a microphone-energy check so it doesn't transcribe
  silence, with a short debounce so half-spoken phrases don't fire early.
- **Speed:** the transformer is **fp8-quantized** (via `torchao`) and run through
  `torch.compile`, and prompt embeddings are cached so unchanged prompts cost
  nothing. Resolution and step count are the main speed dials.
- **LoRAs** are *fused* into the model weights at startup (before quantization), so
  they add zero runtime cost.

This project is a pure-Python replacement for an older ComfyUI workflow that did the
same thing with nodes.

</details>

---

## 📁 Project layout

```
main.py          The app: webcam → restyle → display window + keyboard controls
infer.py         Loads the model, applies LoRAs/fp8/compile, runs each frame
audio.py         Background mic → Whisper → live prompt text
config.py        ⭐ All your settings live here
mic_check.py     Microphone level meter / device lister

INSTALL.bat      One-time: install Python packages
START.bat        Run the app
MIC_CHECK.bat    Run the microphone meter

requirements.txt Python dependencies (INSTALL.bat handles these for you)
models/          Model cache (auto-downloaded; not in git)
```

> 🔧 **For other machines:** the `.bat` files have this computer's paths baked in
> (the Python location and model-cache folder). If you move the project to a
> different PC, open the `.bat` files in Notepad and update those paths.

---

## 🙏 Credits

- **[FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** — Black Forest Labs
- **[🤗 diffusers](https://github.com/huggingface/diffusers)**, **[torchao](https://github.com/pytorch/ao)**, **[PyTorch](https://pytorch.org)**
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — voice transcription
- **Octane Render LoRA** — [civitai.com/models/1883576](https://civitai.com/models/1883576/octane-render)

---

## 📄 License

The code in this repo is yours to license as you wish (MIT recommended). Note that
the **model** (FLUX.2-Klein, Apache-2.0) and any **LoRAs** you add carry their own
licenses — check those before commercial use.
