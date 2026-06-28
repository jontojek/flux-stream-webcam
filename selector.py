# ---------------------------------------------------------------------------
# selector.py  –  fullscreen 1:1 region selector overlay (tkinter)
# ---------------------------------------------------------------------------
# Spawns a semi-transparent fullscreen window on the primary monitor.
# Click and drag to draw a square; the ratio is locked 1:1 (size = max(w, h)).
# Returns (x, y, size) in screen coordinates, or None if cancelled (Esc).
# ---------------------------------------------------------------------------

import tkinter as tk


def select_region() -> tuple[int, int, int] | None:
    """Show fullscreen overlay; user draws a square. Returns (x, y, size) or None."""

    result: list[tuple[int, int, int] | None] = [None]
    start: list[tuple[int, int]] = [(0, 0)]

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-alpha", 0.25)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    root.title("Draw capture region — Esc to cancel")

    canvas = tk.Canvas(root, cursor="crosshair", bg="black", highlightthickness=0)
    canvas.pack(fill=tk.BOTH, expand=True)

    rect_id = canvas.create_rectangle(0, 0, 0, 0, outline="#00FF78", width=2, fill="")
    label_id = canvas.create_text(
        root.winfo_screenwidth() // 2, 40,
        text="Draw a square — release to confirm   |   Esc to cancel",
        fill="white", font=("Consolas", 14),
    )

    def _on_press(event: tk.Event) -> None:
        start[0] = (event.x, event.y)
        canvas.coords(rect_id, event.x, event.y, event.x, event.y)

    def _on_drag(event: tk.Event) -> None:
        x0, y0 = start[0]
        dx = event.x - x0
        dy = event.y - y0
        size = max(abs(dx), abs(dy))
        x1 = x0 + (size if dx >= 0 else -size)
        y1 = y0 + (size if dy >= 0 else -size)
        canvas.coords(rect_id, x0, y0, x1, y1)
        canvas.itemconfigure(label_id, text=f"{size} × {size} px   |   Esc to cancel")

    def _on_release(event: tk.Event) -> None:
        x0, y0 = start[0]
        dx = event.x - x0
        dy = event.y - y0
        size = max(abs(dx), abs(dy))
        if size < 32:
            # Too small — ignore and let user try again
            return
        # Top-left corner in screen coords
        sx = x0 if dx >= 0 else x0 - size
        sy = y0 if dy >= 0 else y0 - size
        result[0] = (sx, sy, size)
        root.destroy()

    def _on_escape(event: tk.Event) -> None:
        result[0] = None
        root.destroy()

    canvas.bind("<ButtonPress-1>", _on_press)
    canvas.bind("<B1-Motion>", _on_drag)
    canvas.bind("<ButtonRelease-1>", _on_release)
    root.bind("<Escape>", _on_escape)

    root.mainloop()
    return result[0]
