# mic_check.py — list audio devices + live microphone level meter.
# Robust: shows ALL devices with host API, never crashes on a bad selection.
#
#   python mic_check.py                # use config.AUDIO_DEVICE / system default
#   python mic_check.py 11             # force device index 11
#   python mic_check.py "Microphone"   # match a device by name (recommended)
import sys, time
import numpy as np
import sounddevice as sd
import config

SR = config.AUDIO_SAMPLE_RATE
THR = config.AUDIO_SILENCE_THRESHOLD

# --- full device dump ------------------------------------------------------
hostapis = sd.query_hostapis()
print("\nAll audio devices (in/out channels, host API):")
inputs = []
for i, d in enumerate(sd.query_devices()):
    ic, oc = d["max_input_channels"], d["max_output_channels"]
    api = hostapis[d["hostapi"]]["name"]
    tag = "  <== INPUT" if ic > 0 else ""
    if ic > 0:
        inputs.append((i, d["name"], api))
    print(f"  [{i:2}] in:{ic} out:{oc}  ({api})  {d['name']}{tag}")

if not inputs:
    print("\n!! No input-capable devices found at all.")
    print("   Your microphone isn't reaching Windows audio. Check:")
    print("   Settings > System > Sound > Input  (device present, not disabled)")
    print("   Settings > Privacy > Microphone    (apps allowed to use mic)")
    print("   and that the mic isn't muted or unplugged.")
    sys.exit(1)

print("\nInput-capable devices:")
for i, name, api in inputs:
    print(f"  [{i}] {name}  ({api})")

# --- resolve selection -----------------------------------------------------
arg = sys.argv[1] if len(sys.argv) > 1 else config.AUDIO_DEVICE
device = None
if arg is not None:
    if isinstance(arg, str) and not arg.isdigit():
        device = arg  # sounddevice matches device by name substring
    else:
        device = int(arg)

try:
    sel = sd.query_devices(device, kind="input")
except Exception as e:
    print(f"\n!! '{arg}' is not a usable input device ({e}).")
    device = inputs[0][0]
    sel = sd.query_devices(device, kind="input")
    print(f"   Falling back to first input: [{device}] {sel['name']}")

print(f"\nUsing input: {sel['name']}")
print(f"Threshold (AUDIO_SILENCE_THRESHOLD) = {THR:.4f}")
print("Talk normally — the bar should jump well past the | marker.  Ctrl+C to stop.\n")

bar_thr = min(int(THR * 300), 59)

def cb(indata, frames, t, status):
    level = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
    n = min(int(level * 300), 60)
    bar = list("#" * n + "-" * (60 - n))
    bar[bar_thr] = "|"
    state = "SPEAK" if level >= THR else "     "
    print(f"\r[{''.join(bar)}] {level:.4f} {state}", end="", flush=True)

try:
    with sd.InputStream(device=device, samplerate=SR, channels=1,
                        dtype="float32", blocksize=int(SR * 0.2), callback=cb):
        while True:
            time.sleep(0.2)
except KeyboardInterrupt:
    print("\nstopped.")
except Exception as e:
    print(f"\n!! Could not open this device: {e}")
    print("   Try a different one from the list above:  MIC_CHECK.bat <index>")
