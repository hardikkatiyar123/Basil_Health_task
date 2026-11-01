# How to reproduce

Setup (Python 3.13, CPU): `python -m venv .venv`, then `pip install -r part1_diagnose/requirements.txt -r part2_transformer/requirements.txt`. The first run downloads PhysioNet EEGBCI to `~/mne_data`.

**Part 1** (each `train.py` run prints its result and appends one line to `part1_diagnose/results.jsonl`):

```bash
cd part1_diagnose
python train.py --channels all --seed N        # N = 0..4; Task 1 (seed 0 after each fix) and 5-seed summary
python train.py --channels frontal --seed N    # Task 3
python train.py --channels central --seed N    # Task 3 control
python erd.py --band mu                        # Task 2 -> erd_mu.png
```

**Part 2** (each seed is printed and appended to `part2_transformer/results.jsonl`):

```bash
cd part2_transformer
python main.py                                 # all models, seeds 0-4, saves figure.png
python main.py --models eegnet                 # or: vanilla / yours (one or more), to run single models
python attention_map.py --seed 0               # attention_topomap.png + top-5 attended electrodes
```

The vanilla transformer needs more than ~4 GB of free RAM; it ran out of memory on this machine (see Part 2 Results).

# Part 1

## Task 1 - Bugs (before/after accuracy + table: #, file:line, what's wrong, effect on accuracy and why)

**Before: 0.667** (seed 0, n_test=84, trial-level split + test-set checkpoint selection, i.e. leaky).
**After all 6 fixes: 0.524 ± 0.061** (mean ± SD over seeds 0-4; best 0.578, worst 0.444; per seed 0.478 / 0.578 / 0.578 / 0.544 / 0.444), n_test=90 (subjects 9-10: 46 hands / 44 feet), **majority-class baseline 0.511**. One-sided binomial test of the mean (47/90 correct) vs p=0.511: **p = 0.46**, not above chance (55/90 = 0.611 would be needed for p<0.05). Seeds share the same 90 test trials, so they are not independent evidence; the test treats the mean as one 90-trial run.
Validation accuracy on subjects 7-8 was 0.698 on average (0.678-0.722) versus 0.524 on subjects 9-10: with 2 held-out people, cross-subject accuracy depends heavily on *which* people are held out, so a single split's number says little about the population (part of the gap is also selection bias, since val picked the checkpoint).

All numbers: `python train.py --seed 0`, cumulative (each row includes the fixes above it).

| # | Fix (file:line) | What was wrong | Effect of the bug (and of the fix) | test_acc after |
|---|---|---|---|---|
| 0 | none (starter) | - | - | 0.667 (n_test=84) |
| 1 | data.py:15 | `RUNS = [5, 9, 13]` are *executed* fists/feet runs; imagery is 6/10/14. | Bug inflated accuracy (↑); fix lowers it (↓). Executed movement gives a stronger ERD plus EMG/movement artefacts, so it is easier to decode than the imagery the task asks about. | 0.619 (n_test=84) |
| 2 | train.py:62-66 | `train_test_split` on *trials* put every subject in train, val and test, so "held-out people" was false. Now split by subject: test=[9,10] (fixed, same as part2), val=[7,8], train=1-6. | Bug inflated accuracy (↑); fix lowers it (↓). With trial-level splits the model sees every test subject in training and can use their idiosyncratic EEG (impedances, head shape, alpha level); across unseen subjects that shortcut is gone. | 0.560 (n_test=84) |
| 3 | train.py:86 (starter) | Checkpoint chosen by *test* accuracy, so the reported number is the max over 30 test evaluations. Now chosen by val; test is evaluated once at the end. | Bug inflated accuracy (↑); fix lowers it (↓). Reporting the max over 30 noisy test evaluations is selection on the test set, which is optimistically biased. | 0.524 (n_test=84; val 0.702 at epoch 7) |
| 4 | train.py:57-60 (starter) | Per-channel z-score mean/std computed on all trials (incl. val/test subjects) before the split. Now fit on train subjects only and applied to val/test. | Bug can inflate accuracy (↑) by leaking test-subject statistics into preprocessing; fix would lower it (↓). Here the leak is label-free (64 means/stds only), so the fix made no measurable difference on seed 0. | 0.524 (n_test=84; val 0.702 at epoch 8) |
| 5 | erd.py:18 | `BASELINE = (0.0, 0.5)` lies *after* the cue, inside the imagery period, where ERD has already started. Now (-1.5, -0.5), pre-cue rest. | No effect on accuracy (erd.py is not used by train.py; re-run confirms 0.524). Bug distorted the ERD plot: a post-cue baseline is already desynchronised (and contains the cue-evoked peak), so ERD is underestimated or flips sign; fix restores a rest reference. | 0.524 (unchanged) |
| 6 | data.py:19 | `TMAX = 5.0`: the last trial of every 125 s run starts at ~120.4 s, so its epoch crosses the run boundary / end of file and MNE silently drops it (2x `BAD boundary`, 1x `TOO_SHORT` per subject): 42 of 45 trials kept, 30/450 trials lost, always the run's final trial. Now `TMAX = 4.0` (still covers the -1.5..-0.5 ERD baseline and the 0-4 s crop). | Bug silently removed 7% of trials, always the same position in the run (no clear direction a priori: less training data ↓, but also a different, exactly balanced test set). Fix restores all 450 trials; seed 0 went down (↓), but one seed on 90 trials is within noise, see the 5-seed summary. | 0.478 (n_test=90; val 0.722 at epoch 22) |

## Task 2 - ERD

`python erd.py --band mu` (8-13 Hz, subjects 1-10, baseline -1.5..-0.5 s) -> `part1_diagnose/erd_mu.png`.
Mean ERD over 0.5-4 s (mean across 10 subjects, computed from the same curves):

| | C3 | Cz | C4 |
|---|---|---|---|
| hands | -13.7% | -4.0% | -8.6% |
| feet | -0.1% | -4.9% | -4.8% |

Hands imagery gives a sustained mu ERD over the lateral hand areas (C3 strongest, C4 weaker), while feet imagery leaves C3/C4 near baseline; this matches the hand knob lying laterally on the precentral gyrus and the foot area lying medially. At Cz the feet ERD is only marginally deeper than hands (-4.9 vs -4.0%), so the expected "feet focal at Cz" effect is barely visible. Cz is weaker than expected because the foot representation sits on the mesial wall inside the interhemispheric fissure, so its generators are deep and tangentially oriented and project poorly to the scalp; Cz also picks up hand-area activity by volume conduction, and foot imagery is harder for naive subjects (between-subject SD ~15-24% dwarfs the class differences, with only ~22 trials/class/subject). The sharp positive peak at ~0.1-0.4 s in every trace (+23 to +44%) is the phase-locked visual evoked response to the cue leaking into 8-13 Hz power, not induced mu activity.

## Task 3 - What is the model using?

`python train.py --channels {all,frontal,central} --seed {0..4}`; val = subjects 7-8, test = subjects 9-10 (n=90 each), majority baseline 0.511, SD with ddof=1.
frontal = Fp1 Fpz Fp2 AF7 AF3 AFz AF4 AF8 F7 F5 F3 F1 Fz F2 F4 F6 F8; central (control, same size) = FC3 FC1 FCz FC2 FC4 C5 C3 C1 Cz C2 C4 C6 CP3 CP1 CPz CP2 CP4.

| channels | val_acc mean ± SD | val best / worst | test_acc mean ± SD | test best / worst | p (test mean vs 0.511, one-sided binomial) |
|---|---|---|---|---|---|
| all (64) | 0.698 ± 0.020 | 0.722 / 0.678 | 0.524 ± 0.061 | 0.578 / 0.444 | 0.46 (47/90) |
| frontal (17) | 0.756 ± 0.035 | 0.811 / 0.722 | 0.564 ± 0.024 | 0.589 / 0.533 | 0.17 (51/90) |
| central (17) | 0.744 ± 0.046 | 0.800 / 0.689 | 0.549 ± 0.051 | 0.622 / 0.500 | 0.30 (49/90) |

Frontal-only (17 electrodes over the forehead, far from motor cortex) reaches val 0.756 ± 0.035, at least as high as all channels (0.698 ± 0.020) and central (0.744 ± 0.046); on the test pair (9-10) all three are at chance. A motor-imagery decoder should not get its best score from the forehead, so the model can, and likely does, exploit a non-brain signal: vertical eye movements (EOG), because the fists target is at the top of the screen and the feet target at the bottom, producing opposite-polarity potentials largest at Fp1/Fp2/AF (blinks/frontalis EMG may add to it). Caveats: both 17-channel subsets beat all 64 channels, so part of the gain is less overfitting from fewer inputs, and the central group's FC row also picks up some EOG; this does not prove the all-channel model uses only the eyes, but it cannot be trusted as a pure motor decoder until ocular activity is removed (ICA/EOG regression, or central cue presentation). Val selected the checkpoint, so val numbers are optimistic, but equally so for all three groups.

## Task 4 - Short answers

1. 58% of 45 = 26 correct. Under H0 p=0.5, P(X>=26) ≈ 0.19, so not above chance; at n=45 you need 29/45 (64%) for p<0.05. Check with a binomial test against the majority-class rate, or a permutation test (shuffle labels, retrain/re-score, see where 58% falls in the null distribution).
2. Spatial filters: for each of the F1 temporal (frequency) filters it learns D weightings across all electrodes, like a learned CSP (e.g. weighting C3/C4 against neighbours to extract mu-band hand-area ERD). `groups=F1` ties each spatial filter to one frequency band and keeps parameters tiny.
3. Present both cues at the same central location (symbol/word at fixation) so gaze doesn't differ by class; in analysis, remove ocular components with ICA/EOG regression or exclude frontal channels and check accuracy holds.

# Part 2

## Hypothesis (written before coding)

Fists-vs-feet information is spatial and sustained (hand-area ERD present vs absent), so attending across 64 electrode tokens with 3D scalp-position encoding will be more stable across seeds than attending across 481 timesteps, and its attention should concentrate on central electrodes (C3/C4/Cz), not frontal ones.

## What I changed and why

Per-timestep tokens (vanilla) make attention model WHEN things happen: 482 tokens, a 482x482 attention matrix per head per layer (O(T²)), no locality prior, 136k params for 8 training subjects. It could not even be trained here: the first batch exceeded ~4 GB free RAM. But Part 1 showed the fists-vs-feet signal is WHERE and sustained: mu ERD over the hand areas for fists (C3 −14%, C4 −9%), near-absent for feet (C3 −0.1%, C4 −5%), over 0.5-4 s. So I tokenised by electrode instead of by time: a shared 8-filter temporal conv per channel, then log mean power (the same quantity ERD measures) gives one 32-d token per electrode. The one inductive bias is the token axis plus scalp geometry: each token gets an MLP encoding of its 3D (x, y, z) position, which encodes anterior-posterior and medial-lateral structure, not just left/right. That matters because the task is bilateral: both hands activate both hemispheres, so the contrast is lateral hand areas vs the medial/vertex foot area. Result: 65 tokens (64 + CLS), a 65x65 attention matrix (~55x smaller), 19k params. Control: the same model with electrode positions shuffled across channels; if real ≈ shuffled, any gain comes from channel tokens/power features, not geometry.

## Results (best / worst / mean across seeds)

Note on the vanilla transformer: The vanilla transformer OOM'd in the first TRAINING batch (forward pass, attention tensor ~238 MB, batch 64 × 4 heads × 482 × 482 floats; several kept for backprop × 3 layers) on this machine with ~4 GB free RAM: the O(T²) cost is a practical limit, not just a theoretical one. Exact error: `DefaultCPUAllocator: not enough memory: you tried to allocate 237899776 bytes`; attempted with seeds 0-4 (twice) and 0-2, each failing in the first batch of seed 0. The channel-token model's attention is 65 × 65 (~55× fewer entries).

`python main.py` (train subjects 1-8, test 9-10, n_test = 90, majority baseline 0.511, seeds 0-4, 40 epochs; SD with ddof=1). "Best-over-epochs" is the max test accuracy over epochs, i.e. selected on the test set; **final-epoch is the honest number**.

| model | params | tokens | attn size (per head, per layer) | best-over-epochs mean / min / max | final-epoch mean ± SD, min, max |
|---|---|---|---|---|---|
| EEGNet | 2,610 | n/a (conv) | n/a | 0.647 / 0.611 / 0.711 | **0.613 ± 0.053**, 0.556, 0.667 |
| Vanilla transformer | 135,746 | 482 | 482×482 | OOM | OOM (first batch of seed 0) |
| Starter YourModel (temporal patches, 32 samples) | 232,898 | 16 | 16×16 | 0.602 / 0.589 / 0.622 | 0.527 ± 0.050, 0.456, 0.578 |
| **Channel tokens + 3D position (real)** | 19,234 | 65 | 65×65 | 0.544 / 0.522 / 0.600 | **0.418 ± 0.047**, 0.367, 0.478 |
| Channel tokens + 3D position (shuffled control) | 19,234 | 65 | 65×65 | 0.553 / 0.533 / 0.589 | 0.464 ± 0.048, 0.389, 0.500 |

The bet failed: final-epoch 0.418 ± 0.047 (best 0.478, worst 0.367) with real positions, below the 0.511 majority baseline on all 5 seeds, and not more stable than the starter (SD 0.047 vs 0.050). Real vs shuffled (0.418 vs 0.464) differ by ~1 SD with n=5, so geometry neither helped nor demonstrably hurt. Attention (seed 0, `python attention_map.py`) went to parietal-occipital electrodes (P7, PO3, PO7, O2), not C3/Cz/C4, falsifying the second prediction; it was weakly selective (top 0.0275 vs uniform 0.0154) and nearly identical for both classes.

Why: (1) the 7-30 Hz band contains both motor mu and occipital alpha at the same frequency; per-channel power tokens can only separate them by location, and occipital alpha is the largest rhythm, tied to visual cue processing and arousal, not imagery. (2) Absolute log-power per electrode carries subject-specific scale (skull, impedance, alpha amplitude), so a rule learned on subjects 1-8 can invert on 9-10, which is consistent with systematically below-chance rather than chance-level scores. EEGNet's depthwise spatial filters learn weighted DIFFERENCES between electrodes, which cancel per-subject gain; that, not attention vs convolution, is the likely reason it wins. (3) Best-over-epochs (0.54-0.65 for every model) hides this: the channel-token model's best-over-epochs averaged 0.544, but its final-epoch score was 0.418, as it fit the training subjects. Lesson: the token axis was a reasonable bet, but token CONTENT must be subject-invariant.

What I'd check next: tokens normalised per trial (log-power minus the trial's mean log-power across electrodes, i.e. a relative scalp topography that removes per-subject gain), or ERD relative to a pre-cue baseline as the token feature; per-epoch test curves to confirm the drift below chance; and the same model with occipital channels removed to test whether visual alpha drives the attention.

80 subjects: the 55x token/memory saving still matters at any data size, but the inductive bias matters less with more data. More subjects would also average over subject-specific scale, so the invariance problem would shrink but not disappear. And because these tokens collapse time into log-power, they discard timing information a larger dataset could exploit.

## Figure

`part2_transformer/figure.png`: t-SNE of the channel-token model's CLS embeddings (seed 0, real positions) for the 90 test trials of subjects 9-10; fists and feet are interleaved within every cluster, so whatever structure the embedding has is not class structure, consistent with the below-chance accuracy. `part2_transformer/attention_topomap.png`: last-layer CLS->electrode attention averaged over test trials and heads, peaking over left parietal-occipital and occipital sites (top-5 marked) with the central strip near or below average.
