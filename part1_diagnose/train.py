"""Train and evaluate a hands-vs-feet decoder on unseen subjects.

Usage:
  python train.py --channels all --seed 0
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn

from data import build_dataset, SUBJECTS
from model import EEGNet, count_params

TEST_SUBJECTS = [9, 10]   # same held-out subjects as part2_transformer
VAL_SUBJECTS = [7, 8]

# Electrode groups for the channel-subset run (Task 3).
# TODO: fill in "frontal" with the electrode names you choose.
CHANNEL_GROUPS = {
    "all": None,
    "frontal": [],
}


def set_seed(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def evaluate(model, X, y, bs=256):
    model.eval(); correct = 0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            xb = torch.from_numpy(X[i:i + bs]); yb = torch.from_numpy(y[i:i + bs])
            correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", choices=list(CHANNEL_GROUPS), default="all")
    ap.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_seed(args.seed)

    X, y, groups, ch_names = build_dataset(args.subjects)
    picks = CHANNEL_GROUPS[args.channels]
    if picks is not None:
        if not picks:
            raise ValueError(f"Fill in CHANNEL_GROUPS['{args.channels}'] first.")
        X = X[:, [ch_names.index(c) for c in picks]]
    print(f"data: {X.shape}, subjects={len(np.unique(groups))}, "
          f"class balance={np.bincount(y)}")

    # Subject-independent evaluation: whole subjects go to test / val / train.
    # Test subjects are fixed across seeds and match part2 (test on 9-10).
    tr_s = [int(s) for s in np.unique(groups) if s not in VAL_SUBJECTS + TEST_SUBJECTS]
    tr = np.flatnonzero(np.isin(groups, tr_s))
    va = np.flatnonzero(np.isin(groups, VAL_SUBJECTS))
    te = np.flatnonzero(np.isin(groups, TEST_SUBJECTS))
    assert not set(groups[tr]) & set(groups[va]) and not set(groups[tr]) & set(groups[te]) \
        and not set(groups[va]) & set(groups[te]), "subject in two splits"
    assert len(tr) and len(va) and len(te), "empty split; check --subjects"
    print(f"subjects: train={tr_s} val={VAL_SUBJECTS} test={TEST_SUBJECTS}")
    te_counts = np.bincount(y[te], minlength=2)
    majority = te_counts.max() / len(te)   # chance level for a constant predictor
    print(f"test class counts (hands, feet)={te_counts.tolist()}, majority baseline={majority:.3f}")

    # Per-channel z-scoring, statistics from TRAIN subjects only.
    mean = X[tr].mean(axis=(0, 2), keepdims=True)
    std = X[tr].std(axis=(0, 2), keepdims=True) + 1e-6
    X = ((X - mean) / std).astype(np.float32)

    n_ch, n_t = X.shape[1], X.shape[2]
    model = EEGNet(n_ch, n_t)
    print(f"EEGNet: {count_params(model)} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    rng = np.random.default_rng(args.seed)

    best_acc, best_state, best_ep = -1.0, None, -1
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(tr)
        for i in range(0, len(order), args.bs):
            b = order[i:i + args.bs]
            xb, yb = torch.from_numpy(X[b]), torch.from_numpy(y[b])
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
        va_acc = evaluate(model, X[va], y[va])
        if va_acc > best_acc:   # keep the best checkpoint (on val; test is untouched)
            best_acc, best_ep = va_acc, ep
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        print(f"epoch {ep:3d}  loss {loss.item():.3f}  val {va_acc:.3f}")

    model.load_state_dict(best_state)
    result = dict(channels=args.channels, seed=args.seed, best_epoch=best_ep,
                  val_acc=best_acc, test_acc=evaluate(model, X[te], y[te]), n_test=len(te),
                  majority=majority)
    print(json.dumps(result))
    with open("results.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
