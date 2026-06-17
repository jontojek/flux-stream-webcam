# ---------------------------------------------------------------------------
# main.py  –  main capture/inference loop (FLUX.2-Klein-4B)
# ---------------------------------------------------------------------------

import os

# Must be set BEFORE torch is imported (triggered by 'from infer import ...').
# Persists the inductor / torch.compile kernel cache to D: so the compiled
# graph survives between runs instead of recompiling every launch.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", r"D:\AI_software\torch_cache")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import time

import cv2
import numpy as np

import config
import interpolate
from audio import AudioThread
from infer import build_pipeline, infer, warmup


def compose_hud(img: np.ndarray, hud: str, prompt_text: str, top_bar: int, bot_bar: int) -> np.ndarray:
    """Pad img with black HUD bars (top/bottom) and draw the status text,
    control hints, and active prompt. Used for both real and RIFE-interpolated
    frames so they look identical in the window."""
    h, w = img.shape[:2]
    out_frame = np.zeros((h + top_bar + bot_bar, w, 3), dtype=np.uint8)
    out_frame[top_bar:top_bar + h, :] = img
    cv2.putText(out_frame, hud, (8, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 120), 1, cv2.LINE_AA)
    cv2.putText(out_frame, "+/- steps  [ ] strength  ,/. feedback  i interp  m/n/r mu  Q quit",
                (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (215, 215, 215), 1, cv2.LINE_AA)
    disp = prompt_text if len(prompt_text) <= 52 else prompt_text[:49] + "..."
    cv2.putText(out_frame, disp, (8, h + top_bar + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out_frame


def open_webcam(device_index: int = 0) -> cv2.VideoCapture:
    """Try DSHOW -> MSMF -> AUTO until the camera opens."""
    backends = [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "AUTO"),
    ]
    for backend, name in backends:
        cap = cv2.VideoCapture(device_index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[main] Webcam opened via {name}.")
                return cap
        cap.release()
    return None


def main() -> None:
    print("[main] Building FLUX.2-Klein pipeline ...")
    pipe = build_pipeline()

    print("[main] Opening webcam ...")
    cap = open_webcam(device_index=0)
    if cap is None:
        print("[main] ERROR: could not open webcam with any backend.")
        print("       Make sure no other app (Chrome, Teams, OBS) has the camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # HUD margins (black bars above/below the image so text never covers it)
    TOP_BAR, BOT_BAR = 44, 32

    WINDOW = "flux-stream-webcam"
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, config.WIDTH, config.HEIGHT + TOP_BAR + BOT_BAR)

    placeholder = np.zeros((config.HEIGHT + TOP_BAR + BOT_BAR, config.WIDTH, 3), dtype=np.uint8)
    msg = "Warming up (fp8 + torch.compile)..." if config.USE_COMPILE else "Warming up..."
    cv2.putText(placeholder, msg, (10, config.HEIGHT // 2 - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(placeholder, "First run compiles kernels (1-3 min).", (10, config.HEIGHT // 2 + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.imshow(WINDOW, placeholder)
    cv2.waitKey(1)

    audio = AudioThread(initial_prompt=config.INITIAL_PROMPT)
    audio.start()

    warmup(pipe, prompt=audio.get_prompt())

    print("\n" + "=" * 60)
    print("  LIVE -- speak to set the style.  Controls:")
    print("    + / -   steps        (quality vs speed)")
    print("    ] / [   strength     (webcam <-> full restyle)")
    print("    . / ,   feedback     (trails / morphing; 0 = off)")
    print("    i       toggle RIFE frame interpolation (smoother, not faster)")
    print("    m / n   schedule-mu raise/lower (flow-shift dial; r = reset to auto)")
    print("    Q/Esc   quit")
    if config.LORAS:
        names = ", ".join(fn for (fn, _w) in config.LORAS)
        print(f"  LoRA (fused): {names}")
        if config.PROMPT_SUFFIX:
            print(f"  Trigger '{config.PROMPT_SUFFIX}' is auto-added to every prompt.")
        print("  Tip: speak a MATERIAL for transformations -- e.g. 'a bronze")
        print("       warrior', 'carved jade', 'molten glass', 'polished marble'.")
        print("  (To change/disable LoRAs: edit LORAS in config.py + restart.)")
    print("  HUD shows: fps  steps  str(ength)  fb(feedback)  lora  interp  mu")
    print("=" * 60 + "\n")

    # Voice-stability debounce state
    active_prompt = audio.get_prompt()
    pending_text  = active_prompt
    pending_since = time.time()
    last_printed  = None
    prev_output   = None   # previous clean styled frame, for temporal feedback

    frame_count = 0
    t0 = time.perf_counter()

    try:
        while True:
            cap.grab()  # flush stale buffer
            ret, frame = cap.read()
            if not ret or frame is None:
                cv2.waitKey(100)
                continue

            # --- Voice debounce: only switch prompt once text holds steady ----
            latest = audio.get_prompt()
            now = time.time()
            if latest != pending_text:
                pending_text = latest
                pending_since = now
            elif latest != active_prompt and (now - pending_since) >= config.VOICE_DEBOUNCE:
                active_prompt = latest

            # --- Temporal feedback: blend live webcam with previous output ---
            cam = cv2.resize(frame, (config.WIDTH, config.HEIGHT))
            if config.FEEDBACK > 0.0 and prev_output is not None:
                src = cv2.addWeighted(cam, 1.0 - config.FEEDBACK,
                                      prev_output, config.FEEDBACK, 0.0)
            else:
                src = cam

            clean_out = np.ascontiguousarray(infer(pipe, src, active_prompt))

            live_fps = frame_count / max(time.perf_counter() - t0, 1e-6)
            hud = (f"{live_fps:4.1f}fps  steps:{config.STEPS}  "
                   f"str:{config.STRENGTH:.2f}  fb:{config.FEEDBACK:.2f}")
            if config.LORAS:
                hud += f"  lora:{config.LORA_LABEL}"
            if config.ENABLE_INTERPOLATION:
                hud += "  interp:on"
            mu_disp = f"{config.SCHEDULE_MU:.2f}" if config.SCHEDULE_MU is not None else "auto"
            hud += f"  mu:{mu_disp}"

            # --- Optional RIFE interpolation: show a blended frame BETWEEN the
            # previous real output and this one before the real one, so the
            # window updates twice per generation (presentation smoothing only --
            # doesn't add new prompt-following content). Toggle live with 'i'.
            if config.ENABLE_INTERPOLATION and prev_output is not None:
                mid = interpolate.interpolate(prev_output, clean_out)
                if mid is not None:
                    cv2.imshow(WINDOW, compose_hud(mid, hud + " [mid]", active_prompt, TOP_BAR, BOT_BAR))
                    cv2.waitKey(1)

            prev_output = clean_out          # feed THIS (clean, no overlay) back next frame

            # --- Compose display: image in the MIDDLE, HUD in black margins ---
            # The generated image is never covered -- text lives only in the bars.
            cv2.imshow(WINDOW, compose_hud(clean_out, hud, active_prompt, TOP_BAR, BOT_BAR))

            frame_count += 1
            if active_prompt != last_printed:
                print(f"\n  PROMPT >> {active_prompt}")
                last_printed = active_prompt
            if frame_count % 15 == 0:
                fps = frame_count / (time.perf_counter() - t0)
                print(f"[main] frame {frame_count:6d}  |  {fps:.2f} FPS  |  "
                      f"steps={config.STEPS} strength={config.STRENGTH:.2f}")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("[main] Quit -- exiting.")
                break
            elif key in (ord("+"), ord("=")):
                config.STEPS = min(config.STEPS + 1, 8)
                print(f"[main] STEPS -> {config.STEPS}")
            elif key in (ord("-"), ord("_")):
                config.STEPS = max(config.STEPS - 1, 1)
                print(f"[main] STEPS -> {config.STEPS}")
            elif key == ord("]"):
                config.STRENGTH = round(min(config.STRENGTH + 0.05, 1.0), 2)
                print(f"[main] STRENGTH -> {config.STRENGTH:.2f}")
            elif key == ord("["):
                config.STRENGTH = round(max(config.STRENGTH - 0.05, 0.40), 2)
                print(f"[main] STRENGTH -> {config.STRENGTH:.2f}")
            elif key == ord("."):
                config.FEEDBACK = round(min(config.FEEDBACK + 0.05, 0.95), 2)
                print(f"[main] FEEDBACK -> {config.FEEDBACK:.2f}   (trails/morphing; , to lower)")
            elif key == ord(","):
                config.FEEDBACK = round(max(config.FEEDBACK - 0.05, 0.0), 2)
                print(f"[main] FEEDBACK -> {config.FEEDBACK:.2f}   ({'off' if config.FEEDBACK == 0 else 'less trails'}; . to raise)")
            elif key == ord("i"):
                config.ENABLE_INTERPOLATION = not config.ENABLE_INTERPOLATION
                state = "ON" if config.ENABLE_INTERPOLATION else "OFF"
                print(f"[main] RIFE interpolation -> {state}")
                if config.ENABLE_INTERPOLATION and not interpolate.available():
                    print("       (RIFE not set up yet -- see README 'Frame interpolation (RIFE)'; "
                          "this will silently no-op until it is.)")
            elif key == ord("m"):
                base = config.SCHEDULE_MU if config.SCHEDULE_MU is not None else config.SCHEDULE_MU_DEFAULT
                config.SCHEDULE_MU = round(min(base + config.SCHEDULE_MU_STEP, config.SCHEDULE_MU_MAX), 2)
                print(f"[main] SCHEDULE_MU -> {config.SCHEDULE_MU:.2f}   (n to lower, r to reset to auto)")
            elif key == ord("n"):
                base = config.SCHEDULE_MU if config.SCHEDULE_MU is not None else config.SCHEDULE_MU_DEFAULT
                config.SCHEDULE_MU = round(max(base - config.SCHEDULE_MU_STEP, config.SCHEDULE_MU_MIN), 2)
                print(f"[main] SCHEDULE_MU -> {config.SCHEDULE_MU:.2f}   (m to raise, r to reset to auto)")
            elif key == ord("r"):
                config.SCHEDULE_MU = None
                print("[main] SCHEDULE_MU -> auto (Klein's resolution-based default)")

            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                print("[main] Window closed -- exiting.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        audio.stop()
        print("[main] Clean shutdown.")


if __name__ == "__main__":
    main()
