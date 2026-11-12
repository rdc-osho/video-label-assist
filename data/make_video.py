#!/usr/bin/env python3
"""Generate a synthetic 'surgical-style' video with known ground truth.

No real patient footage is used. The scene is a textured pink background
(tissue), two instruments that move through three procedure phases, a smoke
blob that partly hides an instrument, and a short blurred stretch (poor view).
Ground truth boxes and phase labels are written to data/ground_truth.csv.

Run: python data/make_video.py
"""
import csv
from pathlib import Path

import cv2
import numpy as np

W, H, FPS, N = 320, 240, 15, 240
SEED = 7
OUT = Path(__file__).parent
rng = np.random.default_rng(SEED)

# Phases (synthetic step names, not clinical definitions)
PHASES = [(0, 80, "approach"), (80, 170, "dissection"), (170, N, "retraction")]
SMOKE = (118, 140)      # frames where smoke hides instrument_1
BLUR = (195, 210)       # frames with a blurred, poor view


def phase_of(f):
    return next(name for a, b, name in PHASES if a <= f < b)


def background():
    base = np.zeros((H, W, 3), np.uint8)
    base[:] = (120, 110, 200)                       # BGR pink
    noise = rng.normal(0, 18, (H // 8, W // 8, 3))
    noise = cv2.resize(noise, (W, H), interpolation=cv2.INTER_CUBIC)
    tex = np.clip(base.astype(float) + noise, 0, 255).astype(np.uint8)
    for _ in range(14):                             # vessels
        p = rng.integers(0, [W, H], size=(4, 2)).astype(np.int32)
        cv2.polylines(tex, [p], False, (70, 60, 170), 2, cv2.LINE_AA)
    return tex


def instrument_path(f, which):
    t = f / N
    if which == 1:   # enters left, works centre, retracts up-left
        x = 40 + 140 * min(1, t * 2.2) - (80 * max(0, t - 0.75) * 4)
        y = 150 + 25 * np.sin(f / 9)
    else:            # enters right, holds, sweeps
        x = 270 - 90 * min(1, t * 1.6) + 30 * np.sin(f / 14)
        y = 80 + 30 * min(1, t * 1.4) + 10 * np.cos(f / 11)
    return float(x), float(y)


def draw_instrument(img, x, y, which):
    """Shaft from the image edge to a tip at (x, y). Box covers the tip region."""
    col = (190, 190, 195) if which == 1 else (150, 160, 170)
    edge = (0, int(y) + 40) if which == 1 else (W, int(y) - 40)
    cv2.line(img, edge, (int(x), int(y)), col, 7, cv2.LINE_AA)
    cv2.circle(img, (int(x), int(y)), 9, (60, 60, 60), -1, cv2.LINE_AA)
    cv2.line(img, (int(x) - 8, int(y) - 8), (int(x) + 8, int(y) + 8), (230, 230, 230), 2)
    return (int(x) - 14, int(y) - 14, 28, 28)        # x, y, w, h


def main():
    bg = background()
    writer = cv2.VideoWriter(str(OUT / "synthetic_procedure.mp4"),
                             cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    rows = []
    for f in range(N):
        frame = bg.copy()
        for which in (1, 2):
            x, y = instrument_path(f, which)
            box = draw_instrument(frame, x, y, which)
            visible = 1
            if which == 1 and SMOKE[0] <= f < SMOKE[1]:
                visible = 0
            cond = "smoke" if not visible else ("blur" if BLUR[0] <= f < BLUR[1] else "clear")
            rows.append({"frame": f, "object": f"instrument_{which}",
                         "x": box[0], "y": box[1], "w": box[2], "h": box[3],
                         "visible": visible, "condition": cond, "phase": phase_of(f)})
        if SMOKE[0] <= f < SMOKE[1]:                 # smoke over instrument 1
            x, y = instrument_path(f, 1)
            overlay = frame.copy()
            cv2.circle(overlay, (int(x), int(y)), 34, (215, 215, 220), -1)
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)
        if BLUR[0] <= f < BLUR[1]:
            frame = cv2.GaussianBlur(frame, (15, 15), 6)
        writer.write(frame)
    writer.release()
    with open(OUT / "ground_truth.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {N} frames and {len(rows)} ground truth labels")


if __name__ == "__main__":
    main()
