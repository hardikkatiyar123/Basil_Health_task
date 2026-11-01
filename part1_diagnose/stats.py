"""Reproduce the summary numbers in ANSWERS.md from results.jsonl (no retraining).

Only post-fix runs are used: lines with a "majority" key, which train.py writes
since all 6 fixes were in. The ERD table reuses erd.py's functions.

Usage: python stats.py
"""
import json
import numpy as np
from scipy.stats import binomtest

from data import load_raw, get_events, EVENT_ID, SUBJECTS
from erd import band_power_epochs, erd_percent, BANDS, PICKS


def run_summaries(path="results.jsonl"):
    runs = [r for r in map(json.loads, open(path)) if "majority" in r]
    print(f"{len(runs)} post-fix runs in {path}")
    for g in ["all", "frontal", "central"]:
        rs = [r for r in runs if r["channels"] == g]
        seeds = [r["seed"] for r in rs]
        if len(set(seeds)) != len(seeds):
            print(f"  WARNING: duplicate seeds for {g}: {seeds}")
        n, maj = rs[0]["n_test"], rs[0]["majority"]
        print(f"\n[{g}] seeds {seeds}, n_test={n}, majority baseline={maj:.3f}")
        for key in ["val_acc", "test_acc"]:
            a = np.array([r[key] for r in rs])
            print(f"  {key}: {a.mean():.3f} ± {a.std(ddof=1):.3f}  best {a.max():.3f}  "
                  f"worst {a.min():.3f}  per seed {[round(float(x), 3) for x in a]}")
        te = np.array([r["test_acc"] for r in rs])
        k = int(round(te.mean() * n))
        p = binomtest(k, n, maj, alternative="greater").pvalue
        k05 = next(j for j in range(n + 1)
                   if binomtest(j, n, maj, alternative="greater").pvalue < 0.05)
        print(f"  one-sided binomial, mean test {k}/{n} vs {maj:.3f}: p = {p:.3f}  "
              f"(p<0.05 needs >= {k05}/{n})")


def task4():
    n = 45
    k = round(0.58 * n)
    p = binomtest(k, n, 0.5, alternative="greater").pvalue
    k05 = next(j for j in range(n + 1)
               if binomtest(j, n, 0.5, alternative="greater").pvalue < 0.05)
    print(f"\n[Task 4.1] 58% of {n} = {k} correct; P(X>={k} | n={n}, p=0.5) = {p:.3f}; "
          f"min k for p<0.05: {k05}/{n} ({k05 / n:.1%})")


def erd_table(band="mu"):
    curves = {c: [] for c in EVENT_ID}
    for s in SUBJECTS:
        raw = load_raw(s)
        ep = band_power_epochs(raw, get_events(raw), BANDS[band])
        for c in EVENT_ID:
            e, t = erd_percent(ep, c)
            curves[c].append(e[[ep.ch_names.index(ch) for ch in PICKS]])
    print(f"\n[Task 2] {band} ERD%, mean over subjects {SUBJECTS[0]}-{SUBJECTS[-1]} (subject SD)")
    for lo, hi in [(0.1, 0.4), (0.5, 4.0)]:
        w = (t >= lo) & (t <= hi)
        for c in EVENT_ID:
            a = np.array(curves[c])[:, :, w].mean(axis=2)          # (subject, channel)
            print(f"  {lo}-{hi} s {c:5s} " + "  ".join(
                f"{ch}: {a[:, i].mean():+6.1f}% ({a[:, i].std():.1f})"
                for i, ch in enumerate(PICKS)))


if __name__ == "__main__":
    run_summaries()
    task4()
    erd_table()
