# Part 1

## Task 1 - Bugs (before/after accuracy + table: #, file:line, what's wrong, effect on accuracy and why)

All numbers: `python train.py --seed 0`, cumulative (each row includes the fixes above it).

| # | Fix (file:line) | What was wrong | Why it moves accuracy | test_acc after |
|---|---|---|---|---|
| 0 | none (starter) | - | - | 0.667 (n_test=84) |
| 1 | data.py:15 | `RUNS = [5, 9, 13]` are *executed* fists/feet runs; imagery is 6/10/14. | Down: real movement gives stronger ERD plus EMG/movement artefacts, so executed trials are easier to classify than imagined ones. | 0.619 (n_test=84) |
| 2 | train.py:62-66 | `train_test_split` on *trials* put every subject in train, val and test, so "held-out people" was false. Now split by subject: test=[9,10] (fixed, same as part2), val=[7,8], train=1-6. | Down: with trial-level splits the model learns each subject's idiosyncratic EEG (impedances, head shape, alpha level) and recognises it at test; across subjects that shortcut is gone. | 0.560 (n_test=84) |
| 3 | train.py:86 (starter) | Checkpoint chosen by *test* accuracy, so the reported number is the max over 30 test evaluations. Now chosen by val; test is evaluated once at the end. | Down: max-over-epochs of a noisy 84-trial test score is optimistically biased (selection on the test set). | 0.524 (n_test=84; val 0.702 at epoch 7) |
| 4 | train.py:57-60 (starter) | Per-channel z-score mean/std computed on all trials (incl. val/test subjects) before the split. Now fit on train subjects only and applied to val/test. | Up (in principle): test-subject statistics leak into preprocessing. Here the leak is label-free, only 64 means/stds, so the effect was negligible: seed 0 unchanged. | 0.524 (n_test=84; val 0.702 at epoch 8) |
| 5 | erd.py:18 | `BASELINE = (0.0, 0.5)` lies *after* the cue, inside the imagery period, where ERD has already started. Now (-1.5, -0.5), pre-cue rest. | No effect on test_acc (erd.py is not used by train.py; re-run confirms 0.524). It changes the ERD plot: a post-cue baseline is already desynchronised, so ERD looks weaker and can even flip sign. | 0.524 (unchanged) |
| 6 | data.py:19 | `TMAX = 5.0`: the last trial of every 125 s run starts at ~120.4 s, so its epoch crosses the run boundary / end of file and MNE silently drops it (2x `BAD boundary`, 1x `TOO_SHORT` per subject): 42 of 45 trials kept, 30/450 trials lost, always the run's final trial. Now `TMAX = 4.0` (still covers the -1.5..-0.5 ERD baseline and the 0-4 s crop). | Direction not predictable a priori: +7% training data, test set grows 84 -> 90 and is no longer exactly balanced. On seed 0 it went down; a single seed on 90 trials is within noise (see 5-seed results). | 0.478 (n_test=90; val 0.722 at epoch 22) |

## Task 2 - ERD

`python erd.py --band mu` (8-13 Hz, subjects 1-10, baseline -1.5..-0.5 s) -> `part1_diagnose/erd_mu.png`.
Mean ERD over 0.5-4 s (mean across 10 subjects, computed from the same curves):

| | C3 | Cz | C4 |
|---|---|---|---|
| hands | -13.7% | -4.0% | -8.6% |
| feet | -0.1% | -4.9% | -4.8% |

Hands imagery gives a sustained mu ERD over the lateral hand areas (C3 strongest, C4 weaker), while feet imagery leaves C3/C4 near baseline; this matches the hand knob lying laterally on the precentral gyrus and the foot area lying medially. At Cz the feet ERD is only marginally deeper than hands (-4.9 vs -4.0%), so the expected "feet focal at Cz" effect is barely visible. Cz is weaker than expected because the foot representation sits on the mesial wall inside the interhemispheric fissure, so its generators are deep and tangentially oriented and project poorly to the scalp; Cz also picks up hand-area activity by volume conduction, and foot imagery is harder for naive subjects (between-subject SD ~15-24% dwarfs the class differences, with only ~22 trials/class/subject). The sharp positive peak at ~0.1-0.4 s in every trace (+23 to +44%) is the phase-locked visual evoked response to the cue leaking into 8-13 Hz power, not induced mu activity.

## Task 3 - What is the model using?

## Task 4 - Short answers

# Part 2

## Hypothesis (written before coding)

## What I changed and why

## Results (best / worst / mean across seeds)

## Figure
