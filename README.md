# 🎨 flux-stream-webcam

> **Speak a style out loud and watch your video feed transform into it — live.**

A real-time, voice-driven video style-transfer app for Windows + NVIDIA GPUs. Talk
into your mic ("a bronze statue", "cyberpunk neon city", "carved jade"), and your
video feed is continuously re-rendered in that style by the **FLUX.2-Klein-4B**
image model. Use your webcam, or capture any window on screen — a UE5 viewport,
a Maya scene, a VLC movie — and style-transfer it in real time.

It's a pure **Python + 🤗 diffusers** app — no ComfyUI, no nodes, just run it.

<!-- TODO: add a screenshot or GIF here, e.g. ![demo](docs/demo.gif) -->

---

## ✨ What it does

- 🎙️ **Voice control** — your speech is transcribed live (Whisper) and becomes the art prompt.
- 🖼️ **Real-time restyle** — every frame is re-imagined by FLUX.2-Klein-4B (~5–14 FPS depending on settings).
- 🖥️ **Live Canvas** — capture any window on your screen (UE5, Maya, VLC, browsers…) instead of a webcam; draw the region with your mouse and swap it live mid-session.
- 🎚️ **Structure dial** — slide from "barely-touched source" to "totally reimagined".
- 🌀 **Temporal feedback** — feed the last frame back in for trails / morphing motion.
- 🧱 **LoRA support** — drop in style/material LoRAs (e.g. an *Octane Render* look) for transformations prompts alone can't do.
- 🎞️ **Frame interpolation (optional)** — RIFE smooths motion between generations for a less choppy feed.
- 🎛️ **Schedule-mu dial** — override the sampler's flow-shift parameter for a sharper or softer look.
- ⚡ **Optimized** — fp8 quantization + `torch.compile` to keep it interactive.

---

## 🖥️ What you need

| | |
|---|---|
| **OS** | Windows 10/11 |
| **GPU** | NVIDIA, **16 GB VRAM or more** (developed on an RTX 5090, 32 GB) |
| **Disk** | ~20 GB free (the model is ~15 GB) |
| **Python** | A working Python 3.11+ install (this project uses Miniconda) |
| **Input** | Webcam + mic, or any app running on your primary monitor |

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

At startup you'll be asked to choose your **input source**:

```
Input source:
  1) Webcam  (default)
  2) Screen region  (UE5, Maya, VLC, ...)
Choice [1]:
```

- **Webcam** — existing webcam path, unchanged.
- **Screen region** — a semi-transparent overlay appears; click and drag to draw a
  square over the window you want to capture. The square is locked 1:1. Press **`Esc`**
  to cancel. Once drawn, the loop starts immediately.

When the output window appears, **start talking**. Say a style and watch it change.

Press **`Q`** (or close the window) to quit.

---

## 🎮 Controls

Everything is shown live on the top bar of the window, so you don't have to memorize it.

| Input | What it does |
|---|---|
| 🎙️ **Speak** | Sets the art style (e.g. "a watercolor sunset") |
| **`+`** / **`-`** | More / fewer diffusion **steps** — higher = nicer but slower |
| **`]`** / **`[`** | **Strength** up/down — how much it restyles vs. keeps the source frame |
| **`.`** / **`,`** | **Feedback** up/down — trails & morphing (0 = off) |
| **`i`** | Toggle RIFE **frame interpolation** on/off (needs one-time setup — see below) |
| **`m`** / **`n`** | **Schedule-mu** up/down — sharper vs. softer look |
| **`r`** | Reset schedule-mu to **auto** (Klein's resolution-based default) |
| **`s`** | Redraw the **screen capture region** (Live Canvas mode only) |
| **`Q`** / **`Esc`** | Quit |

The on-screen status line reads: `fps · steps · str(ength) · fb(feedback) · lora · interp · mu`.

---

## 🖥️ Live Canvas (screen region capture)

When you choose **option 2** at startup, the app captures a square region of your
primary monitor instead of a webcam. This means you can style-transfer:

- A **UE5 viewport** as you fly through a scene
- A **Maya or Blender 3D view**
- A **VLC** or any video player
- A **browser tab**, game, or any other window

### Drawing the region

A fullscreen semi-transparent overlay appears. Click and drag — the square locks to
1:1 automatically (size = `max(width, height)` of your drag). Release the mouse to
confirm. Press **`Esc`** to cancel.

The minimum region size is 32×32 pixels; anything smaller is ignored so you can
try again without restarting.

### Changing the region mid-session

Press **`s`** while the app is running to redraw. The overlay appears again; draw a
new square and the capture region updates instantly with no restart and no
performance cost — it's just four integers passed to the capture call.

> 💡 **Tip:** if your source app is paused or showing a static frame, `dxcam`
> (which uses DXGI desktop duplication) only fires on screen changes. The app will
> skip those frames and hold the last good one rather than hanging.

---

## 🎛️ Tuning it — `config.py`

All the knobs live in **`config.py`** (a plain text file you can edit in Notepad).
The most useful ones:

| Setting | What it means | Tips |
|---|---|---|
| `WIDTH` / `HEIGHT` | Output resolution | **Biggest speed lever.** `512` = best look (~5 FPS), `384` ≈ 9 FPS, `320` ≈ 12–18 FPS. Restart after changing. |
| `STEPS` | Diffusion steps per frame | `1` fastest, `2` good default, `4` max quality. Also live via `+`/`-`. |
| `STRENGTH` | How far it reimagines the source frame (0–1) | `~0.5` = subtle, `~0.9` = full restyle. Also live via `[`/`]`. |
| `FEEDBACK` | Blend in the previous frame (0–1) | `0` off; `0.3–0.6` nice morphing; `0.8+` psychedelic. Also live via `,`/`.`. |
| `WHISPER_MODEL` | Speech recognition model | `"base.en"` (default) or `"tiny.en"` for speed. |
| `AUDIO_DEVICE` | Which microphone to use | `None` = system default, or a name like `"Microphone"` (see below). |
| `LORAS` | List of LoRAs to load | See the LoRA section. |
| `ENABLE_INTERPOLATION` | RIFE frame smoothing | `False` by default. Needs a one-time setup — see **Frame interpolation (RIFE)** below. Also live via `i`. |
| `SCHEDULE_MU` | Sampler flow-shift override | `None` (auto) by default. Live via `m`/`n`/`r`. See **Schedule-mu** below. |

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

## 🎞️ Frame interpolation (RIFE)

RIFE generates a blended frame *between* two real generations, so the window updates
twice as often per generation. It makes motion look smoother — it does **not** make
generation faster or add new prompt-following detail.

### One-time setup

1. Download the **Windows** build from
   **<https://github.com/nihui/rife-ncnn-vulkan/releases>** — the asset named
   `rife-ncnn-vulkan-<date>-windows.zip` (a few MB).
2. Unzip it so you end up with a **`rife\`** folder right next to `main.py`,
   containing `rife-ncnn-vulkan.exe` and model folders (`rife-v4.6\`, etc.).
3. Set `ENABLE_INTERPOLATION = True` in `config.py`, or just press **`i`** while
   the app is running.

No PyTorch/CUDA dependency — it's a portable executable that runs on its own
(Vulkan, works on any GPU vendor).

> 💡 If you forget the setup step, pressing `i` just prints a reminder in the
> console and interpolation silently stays off — it won't crash the app.

---

## 🎛️ Schedule-mu (flow-shift dial)

Klein's flow-matching sampler auto-picks a "mu" time-shift value from your
resolution + step count. This dial overrides it:

- **Lower mu** → steps bunch toward the end → sharper, more detailed, but can lose structure.
- **Higher mu** → steps spread out earlier → softer, smoother, more stable.
- **Auto** (default, `None`) → Klein's own resolution-aware value — the safe choice.

Press **`m`** to raise, **`n`** to lower, **`r`** to reset to auto. The HUD shows
`mu:auto` or the current numeric value. No setup required.

> 💡 Most noticeable at higher `STEPS` — subtle at the default 1–2 steps since
> there's little trajectory left to reshape.

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
| `ModuleNotFoundError: No module named 'numpy'` (or torch, dxcam, etc.) | You ran the wrong Python. Use the `.bat` files, or the full `miniconda3\python.exe` path. |
| Webcam window is black / "could not open webcam" | Another app (Zoom, Teams, OBS, Chrome) is using the camera. Close it. |
| Screen region is black / all one colour | The source app may be using a hardware overlay. Try windowed mode instead of fullscreen. |
| First run hangs for minutes | Normal — it's downloading the model and/or compiling kernels. Only happens once. |
| Out-of-memory / crash on load | Your GPU has too little VRAM. Lower `WIDTH`/`HEIGHT` to `384` or `320` in `config.py`. |
| Voice not picked up | See the **Microphone troubleshooting** section above. |
| It's too slow | Press `-` (fewer steps), or set a lower resolution in `config.py`. |
| `i` does nothing / console says "RIFE not found" | You haven't done the one-time RIFE download yet — see **Frame interpolation (RIFE)** above. |
| `m`/`n`/`r` seem to do nothing visually | Expected at low step counts. Check the HUD's `mu:` readout to confirm the value is changing. |
| `s` key does nothing | Only active in screen-capture mode (option 2 at startup). |

---

## 🧠 How it works (the short version)

<details>
<summary>Click to expand the nerdy bits</summary>

- **Model:** [FLUX.2-Klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
  by Black Forest Labs — a small (4-billion-parameter), distilled image model that
  works in just 1–2 steps, which is what makes real-time possible. Apache-2.0 licensed.
- **The loop:** grab a frame (webcam or screen region) → (optionally blend in the
  last output = *feedback*) → run **img2img** (*strength* controls how far it's
  re-noised) → the spoken prompt steers the style → show the result.
- **Live Canvas:** uses **dxcam** (DXGI desktop duplication — the same API OBS uses),
  which captures hardware-accelerated content that GDI/BitBlt cannot. A tkinter
  fullscreen overlay handles the 1:1 square draw interaction. Changing the region
  mid-session (`s` key) costs nothing — it's just updating four integers.
- **Voice:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) runs on a
  background thread, gated by a microphone-energy check so it doesn't transcribe
  silence, with a short debounce so half-spoken phrases don't fire early.
- **Speed:** the transformer is **fp8-quantized** (via `torchao`) and run through
  `torch.compile`, and prompt embeddings are cached so unchanged prompts cost
  nothing. Resolution and step count are the main speed dials.
- **LoRAs** are *fused* into the model weights at startup (before quantization), so
  they add zero runtime cost.
- **RIFE interpolation** runs as a separate portable executable (`rife-ncnn-vulkan`),
  blending the previous and current output into one extra displayed frame — a
  presentation trick, not a generation-speed trick.
- **Schedule-mu** is a one-line monkeypatch of the internal function Klein's pipeline
  uses to derive its flow-matching scheduler shift (`compute_empirical_mu`), so the
  override needs no fork of `diffusers`.

This project is a pure-Python replacement for an older ComfyUI workflow. RIFE
interpolation and the schedule-mu dial are inspired by knobs in FAL's
[realtime FLUX.2-Klein demo](https://fal.ai/models/fal-ai/flux-2/klein/realtime).

</details>

---

## 📁 Project layout

```
main.py          The app: input → restyle → display window + keyboard controls
capture.py       WebcamCapture / ScreenCapture — unified frame-source abstraction
selector.py      Fullscreen overlay for drawing the Live Canvas region
infer.py         Loads the model, applies LoRAs/fp8/compile, runs each frame
audio.py         Background mic → Whisper → live prompt text
interpolate.py   Optional RIFE frame-interpolation wrapper (presentation smoothing)
config.py        ⭐ All your settings live here
mic_check.py     Microphone level meter / device lister

INSTALL.bat      One-time: install Python packages
START.bat        Run the app
MIC_CHECK.bat    Run the microphone meter

requirements.txt Python dependencies (INSTALL.bat handles these for you)
models/          Model cache (auto-downloaded; not in git)
rife/            RIFE exe + models (manual download; not in git) — see "Frame interpolation"
```

> 🔧 **For other machines:** the `.bat` files have this computer's paths baked in
> (the Python location and model-cache folder). If you move the project to a
> different PC, open the `.bat` files in Notepad and update those paths.

---

## 🙏 Credits

- **[FLUX.2-Klein](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)** — Black Forest Labs
- **[🤗 diffusers](https://github.com/huggingface/diffusers)**, **[torchao](https://github.com/pytorch/ao)**, **[PyTorch](https://pytorch.org)**
- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** — voice transcription
- **[dxcam](https://github.com/ra1nty/DXcam)** — DXGI screen capture
- **Octane Render LoRA** — [civitai.com/models/1883576](https://civitai.com/models/1883576/octane-render)
- **[RIFE](https://github.com/hzwer/arXiv2020-RIFE)** / **[rife-ncnn-vulkan](https://github.com/nihui/rife-ncnn-vulkan)** — optional frame interpolation
- **[FAL's realtime FLUX.2-Klein demo](https://fal.ai/models/fal-ai/flux-2/klein/realtime)** — inspiration for the interpolation and schedule-mu dials

---

## 📄 License

The code in this repo is yours to license as you wish (MIT recommended). Note that
the **model** (FLUX.2-Klein, Apache-2.0) and any **LoRAs** you add carry their own
licenses — check those before commercial use.
