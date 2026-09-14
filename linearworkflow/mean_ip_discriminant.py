"""
Baseline b-tagger: mean lifetimeSignedD0Significance of the top-20 tracks per jet.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

TRAIN_FILE = "mc-flavtag-ttbar-small.h5"
TOP_K      = 20
PLOT_DIR   = "./"

FLAVOUR_TO_LABEL = {5: 0, 4: 1, 0: 2}
LABEL_NAMES      = ["b-jet", "c-jet", "light-jet"]
COLOURS          = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}

print("Loading data...")
with h5py.File(TRAIN_FILE, "r") as f:
    tracks = f["tracks"][:]
    jets   = f["jets"][:]

flavour_id = jets["HadronConeExclTruthLabelID"]
keep       = np.isin(flavour_id, list(FLAVOUR_TO_LABEL.keys()))
tracks, jets = tracks[keep], jets[keep]

valid = tracks["valid"]
ip2d  = tracks["lifetimeSignedD0Significance"].astype(np.float32)
d0    = tracks["lifetimeSignedD0"].astype(np.float32)
good  = valid & (np.abs(d0) < 3.5)
ip2d[~good] = -np.inf

# select top-K tracks per jet by descending IP significance
order     = np.argsort(-ip2d, axis=1)
topk_idx  = order[:, :TOP_K]
topk_ip   = ip2d[np.arange(len(ip2d))[:, None], topk_idx]
topk_good = good[np.arange(len(good))[:, None], topk_idx]

# mean over valid top-K tracks; jets with no valid tracks get 0
n_valid = topk_good.sum(axis=1).clip(min=1)
topk_ip_masked = np.where(topk_good, topk_ip, 0.0)
disc = topk_ip_masked.sum(axis=1) / n_valid   # mean signed IP significance

labels = np.array([FLAVOUR_TO_LABEL[v] for v in jets["HadronConeExclTruthLabelID"]])

# ── discriminant distributions ────────────────────────────────────────
clip = np.percentile(np.abs(disc), 99)
fig, ax = plt.subplots(figsize=(7, 5))
for idx, name in enumerate(LABEL_NAMES):
    ax.hist(disc[labels == idx], bins=80, range=(-clip, clip),
            histtype="step", label=name, color=COLOURS[name],
            linewidth=1.5, density=True)
ax.set_xlabel("mean lifetimeSignedD0Significance (top-20 tracks)")
ax.set_ylabel("Density")
ax.set_title("Mean signed IP significance by flavour")
ax.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR + "mean_ip_discriminant.png", dpi=150, bbox_inches="tight")
print("Saved mean_ip_discriminant.png")

# ── ROC curves ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("ROC — mean signed IP significance (top-20 tracks)", fontweight="bold")
for ax, (bkg_idx, bkg_name) in zip(axes, [(1, "c-jet"), (2, "light-jet")]):
    mask   = (labels == 0) | (labels == bkg_idx)
    y      = (labels[mask] == 0).astype(int)
    fpr, tpr, _ = roc_curve(y, disc[mask])
    ax.plot(tpr, fpr, linewidth=1.5, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.set_xlabel("b-jet efficiency")
    ax.set_ylabel(f"{bkg_name} mistag rate")
    ax.set_title(f"b vs {bkg_name}")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "mean_ip_roc.png", dpi=150, bbox_inches="tight")
print("Saved mean_ip_roc.png")

# ── working points ────────────────────────────────────────────────────
print("\nMistag rates at fixed b-efficiencies:")
b_disc = disc[labels == 0]
for bkg_idx, bkg_name in [(1, "c-jet"), (2, "light-jet")]:
    bkg_disc = disc[labels == bkg_idx]
    print(f"\n  vs {bkg_name}:")
    for eff in [0.70, 0.80, 0.90]:
        thr     = np.percentile(b_disc, 100 * (1 - eff))
        mistag  = (bkg_disc >= thr).mean()
        print(f"    b-eff={eff:.0%}  threshold={thr:.3f}  mistag={mistag:.4f}  rejection={1/mistag:.1f}x")
