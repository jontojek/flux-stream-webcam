# ---------------------------------------------------------------------------
# capture.py  –  unified frame-source abstraction
# ---------------------------------------------------------------------------
# Both classes expose the same minimal interface used by main.py:
#   .grab()            – flush / pre-acquire (no-op for screen capture)
#   .read()            – returns (ok: bool, frame: np.ndarray BGR HWC)
#   .set_region(x, y, size)  – ScreenCapture only; hot-swap capture rect
#   .release()         – clean up
# ---------------------------------------------------------------------------

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Webcam
# ---------------------------------------------------------------------------

class WebcamCapture:
    def __init__(self, device_index: int = 0):
        self._cap = _open_webcam(device_index)
        if self._cap is None:
            raise RuntimeError(
                "Could not open webcam. Make sure no other app (Chrome, Teams, OBS) "
                "has the camera."
            )

    def grab(self) -> None:
        self._cap.grab()

    def read(self) -> tuple[bool, np.ndarray | None]:
        return self._cap.read()

    def release(self) -> None:
        self._cap.release()


def _open_webcam(device_index: int) -> cv2.VideoCapture | None:
    for backend, name in [
        (cv2.CAP_DSHOW, "DSHOW"),
        (cv2.CAP_MSMF,  "MSMF"),
        (cv2.CAP_ANY,   "AUTO"),
    ]:
        cap = cv2.VideoCapture(device_index, backend)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"[capture] Webcam opened via {name}.")
                return cap
        cap.release()
    return None


# ---------------------------------------------------------------------------
# Screen region  (dxcam — DXGI desktop duplication, primary monitor)
# ---------------------------------------------------------------------------

class ScreenCapture:
    """Captures a square region of the primary monitor using dxcam."""

    def __init__(self, x: int, y: int, size: int):
        try:
            import dxcam
        except ImportError:
            raise RuntimeError(
                "dxcam is not installed. Run:  pip install dxcam"
            )
        self._dxcam = dxcam
        self._camera = dxcam.create(device_idx=0, output_idx=0, output_color="BGR")
        self.set_region(x, y, size)

    def set_region(self, x: int, y: int, size: int) -> None:
        self._region = (x, y, x + size, y + size)  # (left, top, right, bottom)
        print(f"[capture] Screen region -> {self._region}")

    def grab(self) -> None:
        pass  # dxcam always returns the latest frame; no pre-grab needed

    def read(self) -> tuple[bool, np.ndarray | None]:
        frame = self._camera.grab(region=self._region)
        if frame is None:
            # dxcam returns None when the frame hasn't changed since last grab
            # (desktop duplication only fires on dirty rects). Return last frame
            # or signal a skip so the caller retries.
            return False, None
        return True, frame  # already BGR via output_color="BGR"

    def release(self) -> None:
        del self._camera
