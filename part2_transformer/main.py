"""
EEG Transformer Sprint — entry point.
Run:  python main.py

This script:
  1. loads the cross-subject split (train subjects 1-8, test 9-10),
  2. trains the two baselines (EEGNet, vanilla transformer),
  3. trains YOUR model across a few seeds and reports best + worst,
  4. saves one figure (figure.png).

>>> YOUR JOB <<<
Implement ONE inductive bias in `YourModel` below (or replace it entirely).
State your hypothesis in the README. Do not quietly edit the vanilla
transformer in models.py — the contrast is the point.
"""
import argparse
import json
import numpy as np
import torch

from data import load_data, get_channel_positions
from models import EEGNet, VanillaEEGTransformer, count_params
from train import train_model, set_seed


# =========================================================================== #
#  YOUR MODEL  — the only part you must write.
#
#  The starter version below is the vanilla transformer with learned TEMPORAL
#  PATCH tokens instead of per-timestep tokens: one concrete, legal example of
#  "inject one inductive bias" (fewer tokens -> a smaller O(T^2) surface to
#  overfit on 8 subjects). Replace its guts with your own idea. Some directions:
#    - structured positional encoding from electrode geometry (get_channel_positions)
#      and/or frequency band,
#    - an attention bias toward scalp neighbourhoods or band-limited windows,
#    - channel (spatial) tokens instead of / in addition to time tokens,
#    - regularisation aimed specifically at the 8-subject data wall.
# =========================================================================== #
import torch.nn as nn


class EncoderLayer(nn.Module):
    """Post-norm transformer layer (same layout as nn.TransformerEncoderLayer)
    that keeps its attention weights so we can see where the model attends."""

    def __init__(self, d_model, n_heads, p):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=p, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(),
                                nn.Dropout(p), nn.Linear(d_model * 2, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p)
        self.attn_weights = None                   # (B, heads, N, N) from last forward

    def forward(self, z):
        a, w = self.attn(z, z, z, need_weights=True, average_attn_weights=False)
        self.attn_weights = w.detach()
        z = self.norm1(z + self.drop(a))
        return self.norm2(z + self.drop(self.ff(z)))


class YourModel(nn.Module):
    """Channel tokens + 3D electrode-position encoding.

    The ONE bias: tokens are electrodes, not timesteps. Each electrode's signal
    goes through a shared temporal conv bank (8 filters), and its log band power
    per filter becomes the token, so attention runs across 64 scalp locations
    (65 tokens with CLS) instead of 481 timesteps. Where a token sits on the
    scalp comes from an MLP on its 3D position; pos_mode="shuffled" permutes
    the positions (fixed seed) as a control for whether geometry matters.
    """

    def __init__(self, n_chan, n_time, n_classes=2, pos=None, pos_mode="real",
                 n_filt=8, kern=64, d_model=32, n_heads=4, depth=2, p=0.3):
        super().__init__()
        self.temporal = nn.Conv1d(1, n_filt, kern, padding="same", bias=False)
        self.feat = nn.Linear(n_filt, d_model)
        pos = np.zeros((n_chan, 3), dtype="float32") if pos is None else pos
        pos = (pos - pos.mean(0)) / pos.std()      # centre, one isotropic scale
        if pos_mode == "shuffled":
            pos = pos[np.random.default_rng(1234).permutation(n_chan)]
        self.register_buffer("xyz", torch.as_tensor(pos, dtype=torch.float32))
        self.pos_mlp = nn.Sequential(nn.Linear(3, d_model), nn.GELU(),
                                     nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls, std=0.02)
        self.layers = nn.ModuleList(EncoderLayer(d_model, n_heads, p) for _ in range(depth))
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x, return_embed=False):   # x: (B, C, T)
        B, C, T = x.shape
        h = self.temporal(x.reshape(B * C, 1, T))            # (B*C, F, T)
        h = torch.log(h.pow(2).mean(-1) + 1e-6)              # log band power (B*C, F)
        tok = self.feat(h.reshape(B, C, -1)) + self.pos_mlp(self.xyz)  # (B, C, d)
        z = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)        # (B, C+1, d)
        for layer in self.layers:
            z = layer(z)
        embed = self.norm(z[:, 0])
        logits = self.head(embed)
        return (logits, embed) if return_embed else logits


# =========================================================================== #


def run_across_seeds(name, build_fn, Xtr, ytr, Xte, yte, seeds, device, **kw):
    """Returns (best_over_epochs, final_epoch) arrays. `best` is the max test acc
    over epochs, i.e. selected on the test set (optimistic); `final` is the
    honest number. Each run is appended to results.jsonl."""
    bests, finals = [], []
    for s in seeds:
        set_seed(s)
        model = build_fn()
        best, final = train_model(model, Xtr, ytr, Xte, yte, device=device, **kw)
        bests.append(best); finals.append(final)
        print(f"  seed {s}: final-epoch test acc {final:.3f}  (best-over-epochs {best:.3f})")
        with open("results.jsonl", "a") as f:
            f.write(json.dumps(dict(model=name, seed=s, final_acc=final,
                                    best_over_epochs=best, n_test=len(yte))) + "\n")
    bests, finals = np.array(bests), np.array(finals)
    print(f"{name}: final-epoch mean {finals.mean():.3f} ± {finals.std(ddof=1) if len(finals) > 1 else 0:.3f}  "
          f"min {finals.min():.3f}  max {finals.max():.3f} | best-over-epochs mean {bests.mean():.3f}  "
          f"min {bests.min():.3f}  max {bests.max():.3f}")
    return bests, finals


def save_figure(model, Xte, yte, device, path="figure.png"):
    """t-SNE of the CLS embeddings on the test subjects, coloured by class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(Xte).to(device), return_embed=True)
        emb = out[1].cpu().numpy() if isinstance(out, tuple) else None
    if emb is None:
        print("  (your model did not return an embedding; skipping figure)")
        return
    n = len(emb)
    emb2 = TSNE(n_components=2, perplexity=min(30, max(5, n // 4)),
                init="pca", random_state=0).fit_transform(emb)
    plt.figure(figsize=(5, 4))
    for c, name in [(0, "fists"), (1, "feet")]:
        m = yte == c
        plt.scatter(emb2[m, 0], emb2[m, 1], s=18, alpha=0.7, label=name)
    plt.legend(); plt.title("Test-subject embeddings (your model)")
    plt.tight_layout(); plt.savefig(path, dpi=130)
    print(f"  saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--models", nargs="+", choices=["eegnet", "vanilla", "yours"],
                    default=["eegnet", "vanilla", "yours"])
    args = ap.parse_args()
    print(f"seeds: {args.seeds}  epochs: {args.epochs}  models: {args.models}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    print("\nloading data (first run downloads ~50 MB)...")
    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    print(f"train {Xtr.shape}  test {Xte.shape}  "
          f"(class balance train {ytr.mean():.2f} / test {yte.mean():.2f}, "
          f"majority baseline {max(yte.mean(), 1 - yte.mean()):.3f})")

    if "eegnet" in args.models:
        print("\n=== Baseline 1: EEGNet (conv priors) ===")
        print(f"params: {count_params(EEGNet(C, T)):,}")
        run_across_seeds("EEGNet", lambda: EEGNet(C, T),
                         Xtr, ytr, Xte, yte, args.seeds, device, epochs=args.epochs)

    if "vanilla" in args.models:
        print("\n=== Baseline 2: vanilla transformer (no EEG prior) ===")
        print(f"params: {count_params(VanillaEEGTransformer(C, T)):,}")
        run_across_seeds("VanillaEEGTransformer", lambda: VanillaEEGTransformer(C, T),
                         Xtr, ytr, Xte, yte, args.seeds, device, epochs=args.epochs)

    if "yours" not in args.models:
        return

    print("\n=== YOUR MODEL: channel tokens + 3D electrode positions ===")
    pos = get_channel_positions()
    print(f"tokens: {C + 1} (64 electrodes + CLS) -> attention {C + 1}x{C + 1} "
          f"vs vanilla {T + 1}x{T + 1}")
    print(f"params: {count_params(YourModel(C, T, pos=pos)):,}")
    for mode in ["real", "shuffled"]:
        print(f"-- pos_mode={mode}")
        run_across_seeds(f"YourModel[pos={mode}]",
                         lambda: YourModel(C, T, pos=pos, pos_mode=mode),
                         Xtr, ytr, Xte, yte, args.seeds, device, epochs=args.epochs)

    print("\nsaving figure...")
    set_seed(0)
    ym = YourModel(C, T, pos=pos)
    train_model(ym, Xtr, ytr, Xte, yte, epochs=args.epochs, device=device)
    save_figure(ym, Xte, yte, device)

    print("\ndone. Report best AND worst in the README — no cherry-picking.")


if __name__ == "__main__":
    main()
