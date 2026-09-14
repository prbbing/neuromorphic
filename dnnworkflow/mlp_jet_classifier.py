"""
Simple MLP jet flavour classifier using signed 2D IP significance.

Input:  top-20 tracks per jet ranked by lifetimeSignedD0Significance
Output: 3-class softmax (0=b, 1=c, 2=light)
"""
import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ── config ────────────────────────────────────────────────────────────
TRAIN_FILE  = "mc-flavtag-ttbar-small.h5"
N_TRAIN     = 150_000
N_TEST      = N_TRAIN // 2
BATCH_SIZE  = 400
EPOCHS      = 100
LR          = 1e-3
HIDDEN      = [40, 40, 40]
TOP_K       = 20
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME  = "mlp_simple.pt"
PLOT_DIR    = "./mlp_simple/"

os.makedirs(PLOT_DIR, exist_ok=True)

FLAVOUR_TO_LABEL = {5: 0, 4: 1, 0: 2}
LABEL_NAMES      = ["b-jet", "c-jet", "light-jet"]
COLOURS          = {"b-jet": "#1f77b4", "c-jet": "#ff7f0e", "light-jet": "#2ca02c"}

# ── data loading ──────────────────────────────────────────────────────
def load_tracks(path, idx=None):
    """Load jets, select top-K tracks by lifetimeSignedD0Significance."""
    with h5py.File(path, "r") as f:
        tracks = f["tracks"][idx] if idx is not None else f["tracks"][:]
        jets   = f["jets"][idx]   if idx is not None else f["jets"][:]

    flavour_id = jets["HadronConeExclTruthLabelID"]
    keep       = np.isin(flavour_id, list(FLAVOUR_TO_LABEL.keys()))
    tracks, jets = tracks[keep], jets[keep]

    valid = tracks["valid"]
    ip2d  = tracks["lifetimeSignedD0Significance"].astype(np.float32)
    d0    = tracks["lifetimeSignedD0"].astype(np.float32)
    good  = valid & (np.abs(d0) < 3.5)
    ip2d[~good] = -np.inf

    order      = np.argsort(-ip2d, axis=1)
    topk_idx   = order[:, :TOP_K]
    topk_feat  = ip2d[np.arange(len(ip2d))[:, None], topk_idx]
    topk_valid = good[np.arange(len(good))[:, None], topk_idx]
    topk_feat  = np.where(topk_valid, topk_feat, 0.0)

    X      = topk_feat.astype(np.float32)                        # (N, TOP_K)
    labels = np.array([FLAVOUR_TO_LABEL[v] for v in jets["HadronConeExclTruthLabelID"]],
                      dtype=np.int64)
    return X, labels


class JetDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self):         return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.y[i]


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, n_classes):
        super().__init__()
        dims   = [in_dim] + hidden
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.BatchNorm1d(dims[i+1]), nn.ReLU()]
        layers += [nn.Linear(dims[-1], n_classes)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


# ── data ──────────────────────────────────────────────────────────────
print("Loading data...")
rng = np.random.default_rng(42)

with h5py.File(TRAIN_FILE, "r") as f:
    all_flavours = f["jets"]["HadronConeExclTruthLabelID"][:]

valid_idx = np.where(np.isin(all_flavours, list(FLAVOUR_TO_LABEL.keys())))[0]
valid_idx = rng.permutation(valid_idx)

test_idx  = np.sort(valid_idx[-N_TEST:])
pool_idx  = valid_idx[:-N_TEST]

n_per_class = N_TRAIN // 3
pool_labels = np.array([FLAVOUR_TO_LABEL[v] for v in all_flavours[pool_idx]])
train_idx   = np.sort(np.concatenate([
    rng.choice(pool_idx[pool_labels == cls], size=n_per_class, replace=False)
    for cls in range(3)
]))

X_train, y_train = load_tracks(TRAIN_FILE, idx=train_idx)
X_test,  y_test  = load_tracks(TRAIN_FILE, idx=test_idx)

print(f"Train — b:{(y_train==0).sum():,}  c:{(y_train==1).sum():,}  light:{(y_train==2).sum():,}")
print(f"Test  — b:{(y_test==0).sum():,}   c:{(y_test==1).sum():,}   light:{(y_test==2).sum():,}")

train_loader = DataLoader(JetDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(JetDataset(X_test,  y_test),  batch_size=BATCH_SIZE)

# ── model ─────────────────────────────────────────────────────────────
model     = MLP(TOP_K, HIDDEN, n_classes=3).to(DEVICE)
optimiser = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
print(f"Device: {DEVICE}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

# ── training ──────────────────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
        optimiser.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(y_b)

    model.eval()
    val_loss, correct = 0.0, 0
    all_preds, all_true, all_probs = [], [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            logits = model(X_b)
            val_loss += criterion(logits, y_b).item() * len(y_b)
            preds = logits.argmax(dim=1)
            correct += (preds == y_b).sum().item()
            all_preds.append(preds.cpu())
            all_true.append(y_b.cpu())
            all_probs.append(torch.softmax(logits, dim=1).cpu())

    history["train_loss"].append(total_loss / len(y_train))
    history["val_loss"].append(val_loss / len(y_test))
    history["val_acc"].append(correct / len(y_test))
    print(f"Epoch {epoch:03d}/{EPOCHS}  "
          f"train_loss={history['train_loss'][-1]:.4f}  "
          f"val_loss={history['val_loss'][-1]:.4f}  "
          f"val_acc={history['val_acc'][-1]:.4f}")

torch.save(model.state_dict(), MODEL_NAME)
print(f"\nSaved {MODEL_NAME}")

# ── evaluation ────────────────────────────────────────────────────────
all_preds = torch.cat(all_preds).numpy()
all_true  = torch.cat(all_true).numpy()
all_probs = torch.cat(all_probs).numpy()

print("\nClassification report:")
print(classification_report(all_true, all_preds, target_names=LABEL_NAMES))
print("Confusion matrix (rows=true, cols=pred):")
print(confusion_matrix(all_true, all_preds))

# ── plots ─────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("MLP jet classifier — training summary", fontweight="bold")
axes[0].plot(epochs, history["train_loss"], label="train")
axes[0].plot(epochs, history["val_loss"],   label="val")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()
axes[1].plot(epochs, history["val_acc"])
axes[1].set_title("Validation accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
cm = confusion_matrix(all_true, all_preds, normalize="true")
im = axes[2].imshow(cm, cmap="Blues", vmin=0, vmax=1)
axes[2].set_xticks([0,1,2]); axes[2].set_yticks([0,1,2])
axes[2].set_xticklabels(["b","c","light"]); axes[2].set_yticklabels(["b","c","light"])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("True")
axes[2].set_title("Confusion matrix (normalised)")
for i in range(3):
    for j in range(3):
        axes[2].text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm[i,j] > 0.5 else "black")
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(PLOT_DIR + "results.png", dpi=150, bbox_inches="tight")
print("Saved results.png")

# discriminant and ROC
pb   = all_probs[:, 0]; pc = all_probs[:, 1]; pu = all_probs[:, 2]
disc = np.log(pb / (0.2 * pc + 0.8 * pu + 1e-10))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"ROC — $\log(p_b\,/\,(0.2\,p_c + 0.8\,p_u))$", fontweight="bold")
for ax, (bkg_idx, bkg_name) in zip(axes, [(1, "c-jet"), (2, "light-jet")]):
    mask   = (all_true == 0) | (all_true == bkg_idx)
    labels = (all_true[mask] == 0).astype(int)
    score  = disc[mask]
    finite = np.isfinite(score)
    fpr, tpr, _ = roc_curve(labels[finite], score[finite])
    ax.plot(tpr, fpr, linewidth=1.5, label=f"AUC = {auc(fpr, tpr):.3f}")
    ax.set_xlabel("b-jet efficiency"); ax.set_ylabel(f"{bkg_name} mistag rate")
    ax.set_title(f"b vs {bkg_name}"); ax.set_yscale("log"); ax.legend()
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(PLOT_DIR + "roc.png", dpi=150, bbox_inches="tight")
print("Saved roc.png")
