"""
Simple spiking CNN (snnTorch) for QCD-vs-ttbar jet classification, using the
per-jet (T, H, W) frame stack produced by bin_events_to_frames.py.

Architecture (kept small/static -- standard Conv2d/Linear + LIF neurons --
specifically so it stays easy to port to hls4ml later):

  input frame (1, 64, 64) at each of T=13 timesteps
    -> Conv2d(1, 8, 5) -> LIF -> MaxPool(2)
    -> Conv2d(8, 16, 5) -> LIF -> MaxPool(2) -> MaxPool(2)
    -> Flatten -> (batch, 576)
    -> concat with bin_widths side vector (batch, 13)
    -> Linear(589, 32) -> LIF
    -> Linear(32, 2) -> LIF
  Output: final membrane potential (continuous) instead of summed spikes,
  giving a smooth score distribution for better ROC discrimination.

Usage:
  python3 train_snn.py frames.npz --epochs 100 --batch-size 256
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from tqdm import tqdm

import snntorch as snn
from snntorch import surrogate


class FrameDataset(Dataset):
    """Lazy dataset — memory-maps frames.npy so only accessed pages are in RAM."""
    def __init__(self, npz_path, indices, label_to_idx):
        npy_path = npz_path.replace(".npz", ".npy")
        lbl_path = npz_path.replace(".npz", "_labels.npy")
        self.frames = np.load(npy_path, mmap_mode="r")
        labels = np.load(lbl_path, allow_pickle=True)
        self.y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)
        self.indices = indices
        # load jet_eta if available (saved alongside frames.npz)
        meta = np.load(npz_path, allow_pickle=True)
        self.jet_eta = meta["jet_eta"].astype(np.float32) if "jet_eta" in meta else None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        x = self.frames[idx].astype(np.float32)  # raw hit counts
        x = x[:, None, :, :]  # (T, 1, H, W)
        eta = float(self.jet_eta[idx]) if self.jet_eta is not None else 0.0
        return torch.from_numpy(x), int(self.y[idx]), eta


class JetSNN(nn.Module):
    def __init__(self, n_bins=64, n_layers=13, n_classes=2, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.lif1  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.lif2  = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(2)
        self.pool4 = nn.MaxPool2d(2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bins, n_bins)
            dummy = self.pool4(self.pool3(self.pool2(self.conv2(self.pool1(self.conv1(dummy))))))
            flat_dim = dummy.numel()

        # fc1 takes conv features + 3 per-timestep scalars: [R_t, bin_width_t, jet_eta]
        self.fc1 = nn.Linear(flat_dim + 3, 32)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(32, n_classes)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x, bin_widths, layer_radii, jet_eta):
        # x:            (batch, T, 1, H, W)
        # bin_widths:   (T,)        physical cm/bin per layer
        # layer_radii:  (T, 1)      physical radius in cm of each layer
        # jet_eta:      (batch, 1)  jet pseudorapidity
        batch, T = x.shape[0], x.shape[1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        for t in range(T):
            cur1 = self.pool1(self.conv1(x[:, t]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool4(self.pool3(self.pool2(self.conv2(spk1))))
            spk2, mem2 = self.lif2(cur2, mem2)

            conv_feat = spk2.flatten(1)                             # (batch, flat_dim)
            r_t  = layer_radii[t].expand(batch, 1)                 # (batch, 1)
            bw_t = bin_widths[t].expand(batch, 1)                  # (batch, 1)
            combined = torch.cat([conv_feat, r_t, bw_t, jet_eta], dim=1)  # (batch, flat_dim+3)

            cur3 = self.fc1(combined)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            _, mem4 = self.lif4(cur4, mem4)

        return mem4  # (batch, 2) final membrane potential — continuous output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", help="frames.npz from bin_events_to_frames.py")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader worker processes for prefetching (default: 0)")
    parser.add_argument("--save-model", default="model.pt",
                        help="Path to save the best model checkpoint (default: model.pt)")
    parser.add_argument("--max-jets", type=int, default=None,
                        help="Limit dataset to this many jets (for quick CPU tests)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # load metadata
    npy_path = args.input_npz.replace(".npz", ".npy")
    lbl_path = args.input_npz.replace(".npz", "_labels.npy")
    bw_path  = args.input_npz.replace(".npz", "_bin_widths.npy")
    frames_mmap = np.load(npy_path, mmap_mode="r")
    labels = np.load(lbl_path, allow_pickle=True)
    bin_widths = np.load(bw_path)  # (T,) physical cm/bin per layer
    n = len(labels)
    label_names = sorted(set(labels.tolist()))
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    n_bins = frames_mmap.shape[-1]
    n_layers = frames_mmap.shape[1]
    print(f"Dataset: {n} jets, canvas {n_bins}x{n_bins}, classes: {label_names}")
    print(f"Bin widths (cm/bin): {np.round(bin_widths, 2)}")

    if args.max_jets is not None:
        n = min(n, args.max_jets)
    idx = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    train_idx_sorted = np.sort(train_idx)
    val_idx_sorted = np.sort(val_idx)

    train_ds = FrameDataset(args.input_npz, train_idx_sorted, label_to_idx)
    val_ds   = FrameDataset(args.input_npz, val_idx_sorted,   label_to_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = len(label_names)
    model = JetSNN(n_bins=n_bins, n_layers=n_layers, n_classes=n_classes).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params}  |  device: {device}")

    # CMS barrel layer radii (cm): PixelBarrel (3), TIB (4), TOB (6)
    cms_radii = np.array([4., 7., 11., 25., 34., 43., 52., 60., 69., 78., 87., 96., 108.],
                         dtype=np.float32)
    assert len(cms_radii) == n_layers, f"Expected {n_layers} radii, got {len(cms_radii)}"
    print(f"Layer radii (cm): {cms_radii}")

    # (T,) tensors — indexed per timestep as bw_tensor[t], radii_tensor[t]
    bw_tensor     = torch.tensor(bin_widths, dtype=torch.float32, device=device)   # (T,)
    radii_tensor  = torch.tensor(cms_radii,  dtype=torch.float32, device=device).unsqueeze(1)  # (T, 1)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        with tqdm(train_loader, desc=f"epoch {epoch+1:3d}/{args.epochs} [train]", leave=False) as pbar:
            for xb, yb, eta in pbar:
                xb, yb = xb.to(device), yb.to(device)
                eta = eta.float().to(device).unsqueeze(1)   # (batch, 1)
                out = model(xb, bw_tensor, radii_tensor, eta)
                loss = loss_fn(out, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(xb)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
        train_loss /= len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb, eta in tqdm(val_loader, desc=f"epoch {epoch+1:3d}/{args.epochs} [val]  ", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                eta = eta.float().to(device).unsqueeze(1)
                pred = model(xb, bw_tensor, radii_tensor, eta).argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch + 1, "model_state_dict": model.state_dict(),
                        "val_acc": val_acc, "n_bins": n_bins, "n_layers": n_layers,
                        "n_classes": n_classes, "label_names": label_names,
                        "bin_widths": bin_widths, "layer_radii": cms_radii}, args.save_model)
            print(f"epoch {epoch+1:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}  [saved]")
        else:
            print(f"epoch {epoch+1:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
