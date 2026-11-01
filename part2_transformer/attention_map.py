"""
Where does the channel-token model attend?
Run:  python attention_map.py [--seed 0] [--pos_mode real]

Trains YourModel exactly as main.py does (same seed, same harness), then
averages the LAST layer's CLS -> electrode attention over test trials and
heads, saves attention_topomap.png and prints the top-5 attended electrodes.
"""
import argparse
import numpy as np
import torch
import mne
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

from data import load_data, get_channel_positions, RUNS
from main import YourModel
from train import train_model, set_seed


def channel_info():
    """mne.Info with names + montage, in the same channel order as X."""
    fnames = eegbci.load_data(1, RUNS[:1], verbose="ERROR")
    raw = concatenate_raws([read_raw_edf(f, preload=False, verbose="ERROR") for f in fnames])
    eegbci.standardize(raw)                      # "Fc5." -> "FC5"; order unchanged
    raw.set_montage(mne.channels.make_standard_montage("standard_1005"),
                    match_case=False, on_missing="ignore")
    return raw.pick(mne.pick_types(raw.info, eeg=True, exclude="bads")).info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--pos_mode", choices=["real", "shuffled"], default="real")
    args = ap.parse_args()
    print(f"seed {args.seed}  pos_mode {args.pos_mode}")

    Xtr, ytr, Xte, yte = load_data()
    C, T = Xtr.shape[1], Xtr.shape[2]
    set_seed(args.seed)
    model = YourModel(C, T, pos=get_channel_positions(), pos_mode=args.pos_mode)
    _, final = train_model(model, Xtr, ytr, Xte, yte, epochs=args.epochs)
    print(f"final-epoch test acc {final:.3f}")

    model.eval()
    with torch.no_grad():
        model(torch.from_numpy(Xte))
    w = model.layers[-1].attn_weights            # (B, heads, 65, 65)
    att = w[:, :, 0, 1:].mean(dim=(0, 1)).numpy()  # CLS -> 64 electrodes
    info = channel_info()
    names = info.ch_names
    top = np.argsort(att)[::-1][:5]
    print("top-5 attended electrodes:",
          ", ".join(f"{names[i]} ({att[i]:.4f})" for i in top),
          f"| uniform would be {1 / (C + 1):.4f}")
    for cls, lab in [(0, "fists"), (1, "feet")]:
        a = w[torch.from_numpy(yte == cls)][:, :, 0, 1:].mean(dim=(0, 1)).numpy()
        print(f"  {lab}: top-5 " + ", ".join(names[i] for i in np.argsort(a)[::-1][:5]))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    mask = np.isin(np.arange(C), top)            # mark the top-5 electrodes
    im, _ = mne.viz.plot_topomap(att, info, axes=ax, show=False, cmap="viridis",
                                 contours=0, mask=mask,
                                 mask_params=dict(markersize=8, markerfacecolor="w"))
    fig.colorbar(im, ax=ax, shrink=0.8, label="CLS attention")
    ax.set_title(f"Last-layer CLS->electrode attention\n(test subjects, seed {args.seed}, pos={args.pos_mode})",
                 fontsize=9)
    fig.tight_layout(); fig.savefig("attention_topomap.png", dpi=130)
    print("saved attention_topomap.png")


if __name__ == "__main__":
    main()
