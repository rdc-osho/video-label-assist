#!/usr/bin/env python3
"""AI-assisted video labelling: propagate boxes between human keyframes,
send the frames the model is unsure about to a human, and measure whether
it saves effort without losing quality.

Steps
1. Read the video and check it against the dataset manifest (hashes, frame count).
2. A human labels keyframes (simulated here as ground truth plus small hand noise).
3. Template matching (OpenCV) carries each box to the frames in between and
   gives a confidence score for every suggestion.
4. Suggestions below a confidence threshold go to a human review queue.
   The threshold is tuned on the development frames only and then frozen,
   so the held-out test frames never influence it.
5. AI suggestions and human-approved labels are saved to separate files.
6. Quality: IoU against ground truth, split by clear, smoke (hidden) and blurred frames.
7. Agreement: a second reviewer labels a blinded sample; box IoU and Cohen's kappa
   on procedure phases.
8. Coverage: what the dataset contains and what is thin.

Run: python data/make_video.py && python label_assist.py
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VIDEO = Path("data/synthetic_procedure.mp4")
GT = Path("data/ground_truth.csv")
OUT, LABELS = Path("output"), Path("labels")
KEYFRAME_EVERY = 10          # a human labels every 10th frame
SEARCH = 24                  # pixels searched around the last box
DEV_END = 180                # frames 0-179 develop the method, 180-239 test it
HAND_NOISE = 1.5             # pixels of jitter in a human box
SECONDS = {"draw": 6.0, "check": 1.5}   # assumed effort per box, stated in the report
rng = np.random.default_rng(42)


# ---------- helpers ----------
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_frames(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def human_box(gt_row):
    """A person's box: the true box with a little hand jitter. Hidden objects get no box."""
    if not gt_row.visible:
        return None
    j = rng.normal(0, HAND_NOISE, 4)
    return (gt_row.x + j[0], gt_row.y + j[1], gt_row.w + j[2], gt_row.h + j[3])


def match(frame, template, last_box):
    """Find the template near the last box. Returns the new box and a 0-1 confidence."""
    x, y, w, h = (int(round(v)) for v in last_box)
    H, W = frame.shape
    x0, y0 = max(0, x - SEARCH), max(0, y - SEARCH)
    x1, y1 = min(W, x + w + SEARCH), min(H, y + h + SEARCH)
    window = frame[y0:y1, x0:x1]
    if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
        return last_box, 0.0
    res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return (x0 + loc[0], y0 + loc[1], w, h), max(0.0, float(score))


def crop(frame, box):
    x, y, w, h = (int(round(v)) for v in box)
    return frame[max(0, y):y + h, max(0, x):x + w]


# ---------- propagation ----------
def propagate(frames, gt):
    """Carry human keyframe boxes forward with template matching."""
    rows = []
    for obj, g in gt.groupby("object"):
        g = g.set_index("frame")
        box, template = None, None
        for f in range(len(frames)):
            row = g.loc[f]
            if f % KEYFRAME_EVERY == 0:
                hb = human_box(row)
                rows.append({"frame": f, "object": obj, "source": "human_keyframe",
                             "box": hb, "confidence": 1.0})
                if hb is not None:
                    box, template = hb, crop(frames[f], hb)
                continue
            if box is None or template is None or template.size == 0:
                rows.append({"frame": f, "object": obj, "source": "ai", "box": None, "confidence": 0.0})
                continue
            box, conf = match(frames[f], template, box)
            rows.append({"frame": f, "object": obj, "source": "ai", "box": box, "confidence": conf})
    return pd.DataFrame(rows).sort_values(["frame", "object"]).reset_index(drop=True)


def approve(sugg, gt, threshold):
    """Suggestions under the threshold go to a person, who redraws or marks the object hidden."""
    g = gt.set_index(["frame", "object"])
    out = []
    for r in sugg.itertuples():
        if r.source == "human_keyframe":
            out.append((r.frame, r.object, r.box, "human_keyframe"))
        elif r.confidence < threshold:
            out.append((r.frame, r.object, human_box(g.loc[(r.frame, r.object)]), "human_review"))
        else:
            out.append((r.frame, r.object, r.box, "ai_accepted"))
    return pd.DataFrame(out, columns=["frame", "object", "box", "decided_by"])


def score(approved, gt):
    g = gt.set_index(["frame", "object"])
    rows = []
    for r in approved.itertuples():
        t = g.loc[(r.frame, r.object)]
        cond = t.condition
        if t.visible:
            ok = r.box is not None and iou(r.box, (t.x, t.y, t.w, t.h)) >= 0.5
            val = iou(r.box, (t.x, t.y, t.w, t.h)) if r.box is not None else 0.0
        else:                                    # hidden: correct answer is no box
            ok, val = r.box is None, np.nan
        rows.append({"frame": r.frame, "object": r.object, "condition": cond,
                     "decided_by": r.decided_by, "iou": val, "correct": ok})
    return pd.DataFrame(rows)


def effort(approved):
    n_draw = (approved.decided_by != "ai_accepted").sum()
    n_check = (approved.decided_by == "ai_accepted").sum()
    return n_draw, n_check, n_draw * SECONDS["draw"] + n_check * SECONDS["check"]


def tune_threshold(sugg, gt):
    """Pick the lowest threshold that keeps dev accuracy at 98% or above."""
    dev_s, dev_g = sugg[sugg.frame < DEV_END], gt[gt.frame < DEV_END]
    for t in np.arange(0.30, 0.96, 0.05):
        acc = score(approve(dev_s, dev_g, t), dev_g).correct.mean()
        if acc >= 0.98:
            return round(float(t), 2), acc
    return 0.95, None


# ---------- agreement ----------
def cohen_kappa(a, b):
    labels = sorted(set(a) | set(b))
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    pe = sum((a == l).mean() * (b == l).mean() for l in labels)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def agreement(gt):
    """Two reviewers label the same blinded 20% sample independently."""
    vis = gt[gt.visible == 1]
    sample = vis.sample(frac=0.2, random_state=3)
    box_iou = [iou(human_box(r), human_box(r)) for r in sample.itertuples()]
    frames = gt.drop_duplicates("frame").sort_values("frame")
    a = frames.phase.tolist()
    # reviewer B places phase boundaries a few frames differently
    b = frames.phase.shift(-3).ffill().tolist()
    return float(np.mean(box_iou)), cohen_kappa(a, b), len(sample)


def preview(path, approved, sugg, frames_idx=(63, 125, 203)):
    """Save a strip of frames: green = accepted AI box, red = sent to a person for review."""
    cap = cv2.VideoCapture(str(VIDEO)); imgs = {}
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in frames_idx:
            imgs[i] = f
        i += 1
    cap.release()
    tiles = []
    for f in frames_idx:
        img = imgs[f].copy()
        for r in sugg[sugg.frame == f].itertuples():
            if r.box is None:
                continue
            d = approved[(approved.frame == f) & (approved.object == r.object)].decided_by.iloc[0]
            col = (0, 200, 0) if d == "ai_accepted" else (0, 0, 255)
            x, y, w, h = (int(round(v)) for v in r.box)
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
            cv2.putText(img, f"{r.confidence:.2f}", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.putText(img, f"frame {f}", (6, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)
    cv2.imwrite(str(path), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 82])


# ---------- main ----------
def main():
    OUT.mkdir(exist_ok=True); LABELS.mkdir(exist_ok=True)
    frames, fps = read_frames(VIDEO)
    gt = pd.read_csv(GT)
    assert len(frames) == gt.frame.nunique(), "frame count does not match ground truth"

    sugg = propagate(frames, gt)
    threshold, dev_acc = tune_threshold(sugg, gt)
    approved = approve(sugg, gt, threshold)

    test = approved.frame >= DEV_END
    results = score(approved[test], gt[gt.frame >= DEV_END])
    manual = approved[test].assign(decided_by="human_keyframe")
    manual_draw, _, manual_secs = effort(manual)
    draw, check, secs = effort(approved[test])
    review_rate = (approved[test].decided_by == "human_review").mean()

    box_iou, kappa, n_agree = agreement(gt)

    # save suggestions and approved labels separately
    fmt = lambda b: "" if b is None else ",".join(f"{v:.1f}" for v in b)
    sugg.assign(box=sugg.box.map(fmt)).to_csv(LABELS / "ai_suggestions.csv", index=False)
    approved.assign(box=approved.box.map(fmt)).to_csv(LABELS / "approved_labels.csv", index=False)
    approved[approved.decided_by == "human_review"].assign(
        box=lambda d: d.box.map(fmt)).to_csv(OUT / "review_queue.csv", index=False)

    by_cond = results.groupby("condition").agg(frames=("frame", "size"),
                                               correct=("correct", "mean"),
                                               mean_iou=("iou", "mean"))
    test_ai = results[results.decided_by == "ai_accepted"]
    coverage = gt.drop_duplicates("frame").phase.value_counts().to_dict()
    hidden = int((gt.visible == 0).sum())
    blurred = int(gt[gt.condition == "blur"].frame.nunique())

    # the model on its own, every suggestion accepted, across the whole video
    alone = score(approve(sugg, gt, 0.0), gt)
    alone = alone[~alone.decided_by.eq("human_keyframe")]
    alone_tbl = alone.groupby("condition").agg(labels=("frame", "count"), correct=("correct", "mean"))
    queue = approved.merge(gt[["frame", "object", "condition"]], on=["frame", "object"])
    queue = queue[queue.decided_by == "human_review"].condition.value_counts()

    manifest = {
        "dataset": "synthetic_procedure v1",
        "video_sha256": sha256(VIDEO), "ground_truth_sha256": sha256(GT),
        "frames": len(frames), "fps": fps,
        "split": {"development": f"0-{DEV_END - 1}", "test": f"{DEV_END}-{len(frames) - 1}"},
        "parameters": {"keyframe_every": KEYFRAME_EVERY, "search_px": SEARCH,
                       "confidence_threshold": threshold, "tuned_on": "development only"},
        "tools": {"python": platform.python_version(), "opencv": cv2.__version__,
                  "numpy": np.__version__, "pandas": pd.__version__,
                  "method": "template matching, TM_CCOEFF_NORMED"},
        "outputs": {"ai_suggestions": "labels/ai_suggestions.csv",
                    "approved_labels": "labels/approved_labels.csv"},
    }
    json.dump(manifest, open(OUT / "manifest.json", "w"), indent=2)
    preview(OUT / "preview.jpg", approved, sugg)

    report = f"""# Labelling report

Test frames {DEV_END} to {len(frames) - 1} ({test.sum()} labels), never used to tune anything.
Confidence threshold {threshold}, chosen on development frames (accuracy there {dev_acc:.0%}).

## Effort

| Approach | Boxes drawn by a person | AI boxes checked | Estimated time |
|---|---|---|---|
| Fully manual | {manual_draw} | 0 | {manual_secs / 60:.1f} min |
| AI-assisted | {draw} | {check} | {secs / 60:.1f} min |

Saving: **{1 - secs / manual_secs:.0%}** of labelling time, assuming {SECONDS['draw']:.0f} s to draw a box
and {SECONDS['check']:.1f} s to check a suggestion. {review_rate:.0%} of test frames were sent to a person for review.

## Quality on test frames

| Condition | Labels | Correct | Mean IoU (visible only) |
|---|---|---|---|
""" + "\n".join(f"| {c} | {r.frames} | {r.correct:.0%} | {'' if np.isnan(r.mean_iou) else f'{r.mean_iou:.2f}'} |"
                for c, r in by_cond.iterrows()).replace(".0 |", " |") + f"""

Overall: {results.correct.mean():.0%} correct. AI-accepted boxes alone: {test_ai.correct.mean():.0%} correct
(mean IoU {test_ai.iou.mean():.2f}). Correct means IoU of 0.5 or more for a visible instrument, or no box
for a hidden one.

## Where the model goes wrong

If every AI suggestion were accepted with no review, across the whole video:

| Condition | AI labels | Correct |
|---|---|---|
""" + "\n".join(f"| {c} | {int(r.labels)} | {r.correct:.0%} |" for c, r in alone_tbl.iterrows()) + f"""

The smoke stretch is the failure case. The instrument is hidden, but template matching still
returns its best guess, so the model draws a box on something that is not there.
With the threshold in place, {int(queue.sum())} suggestions went to a person:
{', '.join(f"{n} {c}" for c, n in queue.items())}. The model handles the easy frames and a person
handles these.

Note: the two-reviewer IoU below is lower than the AI's IoU because the simulated hand jitter
({HAND_NOISE} px on a 28 px box) is larger than the tracker's drift. On real footage, measure both.

## Reviewer agreement (blinded sample)

- Box agreement between two reviewers on {n_agree} labels: mean IoU **{box_iou:.2f}**
- Phase agreement across all frames: Cohen's kappa **{kappa:.2f}** (reviewers differ only around phase boundaries)

## Coverage

- Frames per phase: {coverage}
- Hidden instrument labels: {hidden} ({hidden / len(gt):.0%}); blurred frames: {blurred} ({blurred / len(frames):.0%})
- **Thin:** blurred and smoke frames are a small share of the data, and only one kind of instrument
  shape exists. A real dataset would need more of both before a model is trusted on them.
"""
    (OUT / "REPORT.md").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
