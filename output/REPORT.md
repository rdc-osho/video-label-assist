# Labelling report

Test frames 180 to 239 (120 labels), never used to tune anything.
Confidence threshold 0.9, chosen on development frames (accuracy there 100%).

## Effort

| Approach | Boxes drawn by a person | AI boxes checked | Estimated time |
|---|---|---|---|
| Fully manual | 120 | 0 | 12.0 min |
| AI-assisted | 43 | 77 | 6.2 min |

Saving: **48%** of labelling time, assuming 6 s to draw a box
and 1.5 s to check a suggestion. 26% of test frames were sent to a person for review.

## Quality on test frames

| Condition | Labels | Correct | Mean IoU (visible only) |
|---|---|---|---|
| blur | 30 | 100% | 0.85 |
| clear | 90 | 100% | 0.85 |

Overall: 100% correct. AI-accepted boxes alone: 100% correct
(mean IoU 0.87). Correct means IoU of 0.5 or more for a visible instrument, or no box
for a hidden one.

## Where the model goes wrong

If every AI suggestion were accepted with no review, across the whole video:

| Condition | AI labels | Correct |
|---|---|---|
| blur | 28 | 100% |
| clear | 384 | 100% |
| smoke | 20 | 0% |

The smoke stretch is the failure case. The instrument is hidden, but template matching still
returns its best guess, so the model draws a box on something that is not there.
With the threshold in place, 80 suggestions went to a person:
47 clear, 20 smoke, 13 blur. The model handles the easy frames and a person
handles these.

Note: the two-reviewer IoU below is lower than the AI's IoU because the simulated hand jitter
(1.5 px on a 28 px box) is larger than the tracker's drift. On real footage, measure both.

## Reviewer agreement (blinded sample)

- Box agreement between two reviewers on 92 labels: mean IoU **0.76**
- Phase agreement across all frames: Cohen's kappa **0.96** (reviewers differ only around phase boundaries)

## Coverage

- Frames per phase: {'dissection': 90, 'approach': 80, 'retraction': 70}
- Hidden instrument labels: 22 (5%); blurred frames: 15 (6%)
- **Thin:** blurred and smoke frames are a small share of the data, and only one kind of instrument
  shape exists. A real dataset would need more of both before a model is trusted on them.
