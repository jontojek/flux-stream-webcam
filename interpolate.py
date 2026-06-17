# ---------------------------------------------------------------------------
# interpolate.py  –  optional RIFE frame interpolation (presentation smoother)
# ---------------------------------------------------------------------------
# Wraps the standalone rife-ncnn-vulkan.exe (portable, no PyTorch/CUDA needed).
# Given two real generated frames, produces one AI-blended frame "between"
# them, so main.py's display loop can show [interpolated, real] instead of
# just [real] each cycle -- roughly doubling the PERCEIVED frame rate.
#
# This does NOT speed up generation and does NOT add new prompt-following
# content -- it's pure motion smoothing between two real frames. It's the
# same trick FAL's realtime Klein demo uses (their `enable_interpolation`
# flag): https://fal.ai/models/fal-ai/flux-2/klein/realtime
#
# ---- One-time setup (this part can't be automated -- it's a binary) -------
#   1. Download the Windows build from:
#        https://github.com/nihui/rife-ncnn-vulkan/releases
#      (the asset named rife-ncnn-vulkan-<date>-windows.zip)
#   2. Unzip it so you end up with a `rife\` folder next to main.py
#      containing rife-ncnn-vulkan.exe and model folders (rife-v4.6\, etc.)
#   3. Set ENABLE_INTERPOLATION = True in config.py (or press `i` live).
# See the README's "Frame interpolation (RIFE)" section for the full version
# of these steps.
# ---------------------------------------------------------------------------

import os
import subprocess
import tempfile

import cv2
import numpy as np

import config

_warned = False  # only print the "not set up" message once per run


def available() -> bool:
    """True if the RIFE exe + model folder actually exist on disk."""
    global _warned
    ok = os.path.isfile(config.RIFE_EXE) and os.path.isdir(config.RIFE_MODEL)
    if not ok and not _warned:
        print(f"[interpolate] RIFE not found (looked for {config.RIFE_EXE}). "
              f"Interpolation will stay off -- see README 'Frame interpolation (RIFE)'.")
        _warned = True
    return ok


def interpolate(frame_a: np.ndarray, frame_b: np.ndarray) -> np.ndarray | None:
    """Return one AI-blended frame "between" frame_a and frame_b (BGR uint8
    arrays, same shape). Returns None if RIFE isn't set up or the call fails
    -- callers should treat None as "just skip showing an extra frame"."""
    if not available():
        return None

    with tempfile.TemporaryDirectory() as tmp:
        path_a = os.path.join(tmp, "a.png")
        path_b = os.path.join(tmp, "b.png")
        path_o = os.path.join(tmp, "o.png")
        cv2.imwrite(path_a, frame_a)
        cv2.imwrite(path_b, frame_b)
        try:
            subprocess.run(
                [config.RIFE_EXE, "-0", path_a, "-1", path_b, "-o", path_o,
                 "-m", config.RIFE_MODEL],
                check=True, capture_output=True, timeout=2.0,
            )
            mid = cv2.imread(path_o)
            if mid is None:
                print("[interpolate] RIFE ran but produced no output frame.")
            return mid
        except subprocess.TimeoutExpired:
            print("[interpolate] RIFE call timed out (>2s) -- skipping this frame.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"[interpolate] RIFE exited with an error: {e.stderr.decode(errors='ignore')[:200]}")
            return None
        except Exception as e:
            print(f"[interpolate] RIFE call failed: {e}")
            return None
