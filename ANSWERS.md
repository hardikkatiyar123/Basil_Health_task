# Part 1

## Task 1 - Bugs (before/after accuracy + table: #, file:line, what's wrong, effect on accuracy and why)

All numbers: `python train.py --seed 0`, cumulative (each row includes the fixes above it).

| # | Fix (file:line) | What was wrong | Why it moves accuracy | test_acc after |
|---|---|---|---|---|
| 0 | none (starter) | - | - | 0.667 (n_test=84) |
| 1 | data.py:15 | `RUNS = [5, 9, 13]` are *executed* fists/feet runs; imagery is 6/10/14. | Down: real movement gives stronger ERD plus EMG/movement artefacts, so executed trials are easier to classify than imagined ones. | 0.619 (n_test=84) |
| 2 | train.py:62-66 | `train_test_split` on *trials* put every subject in train, val and test, so "held-out people" was false. Now split by subject: test=[9,10] (fixed, same as part2), val=[7,8], train=1-6. | Down: with trial-level splits the model learns each subject's idiosyncratic EEG (impedances, head shape, alpha level) and recognises it at test; across subjects that shortcut is gone. | 0.560 (n_test=84) |
| 3 | train.py:86 (starter) | Checkpoint chosen by *test* accuracy, so the reported number is the max over 30 test evaluations. Now chosen by val; test is evaluated once at the end. | Down: max-over-epochs of a noisy 84-trial test score is optimistically biased (selection on the test set). | 0.524 (n_test=84; val 0.702 at epoch 7) |

## Task 2 - ERD

## Task 3 - What is the model using?

## Task 4 - Short answers

# Part 2

## Hypothesis (written before coding)

## What I changed and why

## Results (best / worst / mean across seeds)

## Figure
