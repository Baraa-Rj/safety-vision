"""Train the fall classifier with a scene-level split and SR2.4-aware eval.

Splitting is by source video (not by window): windows from one clip are highly
correlated, so a random split would leak and inflate metrics — the same issue
flagged in the PPE pipeline. Held-out videos give an honest sensitivity / FPR.

Usage:
    python train.py --data dataset.npz --out fall_model.pt \
        --val-frac 0.3 --epochs 60 --fpr-target 0.07
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from model import FallTCN, save_fall_model, standardize


def scene_split(groups, val_frac, seed):
    vids = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(vids)
    n_val = max(1, int(round(len(vids) * val_frac)))
    val_vids = set(vids[:n_val].tolist())
    val_mask = np.array([g in val_vids for g in groups])
    return ~val_mask, val_mask, sorted(val_vids)


def metrics(y_true, prob, thr):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0          # recall on falls
    fpr = fp / (fp + tn) if fp + tn else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    return dict(thr=thr, sens=sens, fpr=fpr, prec=prec, tp=tp, fp=fp, tn=tn, fn=fn)


def pick_threshold(y_true, prob, fpr_target):
    """Highest sensitivity among thresholds whose FPR <= target; else best F1."""
    best = None
    for thr in np.linspace(0.05, 0.95, 91):
        m = metrics(y_true, prob, thr)
        if m["fpr"] <= fpr_target:
            if best is None or m["sens"] > best["sens"]:
                best = m
    if best is not None:
        return best
    # fallback: maximize F1 if the FPR target is unreachable on val
    f1s = []
    for thr in np.linspace(0.05, 0.95, 91):
        m = metrics(y_true, prob, thr)
        f1 = 2 * m["prec"] * m["sens"] / (m["prec"] + m["sens"] + 1e-9)
        f1s.append((f1, m))
    return max(f1s, key=lambda t: t[0])[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dataset.npz")
    ap.add_argument("--out", default="fall_model.pt")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--fpr-target", type=float, default=0.07)   # SR2.4
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = np.load(args.data, allow_pickle=True)
    X, y, groups = d["X"], d["y"], d["groups"]
    window, n_features = int(d["window"]), int(d["n_features"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tr, va, val_vids = scene_split(groups, args.val_frac, args.seed)
    if y[tr].sum() == 0 or y[va].sum() == 0:
        raise SystemExit("Split has no fall windows on one side; add more fall "
                         "videos or adjust --val-frac / --seed.")
    print(f"train videos: {len(set(groups[tr]))}  val videos: {val_vids}")
    print(f"train windows: {tr.sum()} ({int(y[tr].sum())} fall)  "
          f"val windows: {va.sum()} ({int(y[va].sum())} fall)")

    mean = X[tr].reshape(-1, n_features).mean(0)
    std = X[tr].reshape(-1, n_features).std(0)
    Xtr = standardize(X[tr], mean, std)
    Xva = standardize(X[va], mean, std)

    # drop_last avoids a final batch of size 1, which makes BatchNorm1d raise
    # "Expected more than 1 value per channel". Only drop when there's at least
    # one full batch, so a small training set still trains.
    dl = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(y[tr])),
        batch_size=args.batch, shuffle=True,
        drop_last=int(tr.sum()) > args.batch,
    )

    counts = np.bincount(y[tr], minlength=2).astype(np.float32)
    w = counts.sum() / (2.0 * np.clip(counts, 1, None))
    model = FallTCN(n_features=n_features, hidden=args.hidden).to(device)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(w, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    Xva_t = torch.from_numpy(Xva).to(device)
    best_sens, best_state, best_m = -1.0, None, None
    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = crit(model(xb.to(device)), yb.to(device))
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.softmax(model(Xva_t), 1)[:, 1].cpu().numpy()
        m = pick_threshold(y[va], prob, args.fpr_target)
        if m["sens"] > best_sens:
            best_sens = m["sens"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_m = m
        if ep % 10 == 0 or ep == args.epochs:
            print(f"ep {ep:3d}  loss {loss.item():.3f}  "
                  f"val sens {m['sens']:.3f}  fpr {m['fpr']:.3f}  thr {m['thr']:.2f}")

    model.load_state_dict(best_state)
    save_fall_model(args.out, model, mean, std, window, n_features, best_m["thr"])
    print("\nBest val (held-out scenes):")
    print(f"  sensitivity {best_m['sens']:.3f}  window-FPR {best_m['fpr']:.3f}  "
          f"precision {best_m['prec']:.3f}  threshold {best_m['thr']:.2f}")
    print(f"  confusion  TP {best_m['tp']}  FP {best_m['fp']}  "
          f"TN {best_m['tn']}  FN {best_m['fn']}")
    print(f"  SR2.4 target: sensitivity >= 0.85, FPR <= {args.fpr_target}")
    print("  NOTE: window-FPR is per-window, a conservative proxy. The deployed "
          "SR2.4 metric is FALSE ALARMS PER HOUR after the confirm-streak + "
          "cooldown gate — validate that on real negative footage.")
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
