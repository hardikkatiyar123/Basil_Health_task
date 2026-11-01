# Part 1

## Task 1 - Bugs (before/after accuracy + table: #, file:line, what's wrong, effect on accuracy and why)

All numbers: `python train.py --seed 0`, cumulative (each row includes the fixes above it).

| # | Fix (file:line) | What was wrong | Why it moves accuracy | test_acc after |
|---|---|---|---|---|
| 0 | none (starter) | - | - | 0.667 (n_test=84) |
| 1 | data.py:15 | `RUNS = [5, 9, 13]` are *executed* fists/feet runs; imagery is 6/10/14. | Down: real movement gives stronger ERD plus EMG/movement artefacts, so executed trials are easier to classify than imagined ones. | 0.619 (n_test=84) |

## Task 2 - ERD

## Task 3 - What is the model using?

## Task 4 - Short answers

# Part 2

## Hypothesis (written before coding)

## What I changed and why

## Results (best / worst / mean across seeds)

## Figure
