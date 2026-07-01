"""
Simple spiking CNN (snnTorch) for QCD-vs-ttbar jet classification, using the
per-jet (T, H, W) frame stack produced by bin_events_to_frames.py.

Architecture (kept small/static -- standard Conv2d/Linear + LIF neurons --
specifically so it stays easy to port to hls4ml later):

  input frame (1, 32, 32) at each of T=13 timesteps
    -> Conv2d(1, 8, 5) -> LIF -> MaxPool(2)
    -> Conv2d(8, 16, 5) -> LIF -> MaxPool(2)
    -> Flatten -> Linear(16*5*5, 32) -> LIF
    -> Linear(32, 2)  (readout, accumulated over time)

Each timestep's frame is binarized (hit / no-hit per pixel) and fed in as
input current at that step; spikes propagate through the network across all
13 timesteps, and the final classification uses the summed (rate-coded)
output spikes / membrane potential of the last layer.

Usage:
  python3 train_snn.py frames.npz --epochs 30
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import snntorch as snn
from snntorch import surrogate


class JetSNN(nn.Module):
    def __init__(self, n_bins=32, beta=0.9):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=5)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.pool2 = nn.MaxPool2d(2)
        # extra parameter-free pooling stage just to shrink the flattened
        # feature map (and hence fc1's size) before the FC layer -- fc1 is
        # by far the largest contributor to the parameter count otherwise
        self.pool3 = nn.MaxPool2d(2)

        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_bins, n_bins)
            dummy = self.pool3(self.pool2(self.conv2(self.pool1(self.conv1(dummy)))))
            flat_dim = dummy.numel()
        self.fc1 = nn.Linear(flat_dim, 32)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        self.fc2 = nn.Linear(32, 2)
        self.lif4 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x: (batch, T, 1, H, W)
        batch, T = x.shape[0], x.shape[1]
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        out_spk_sum = torch.zeros(batch, 2, device=x.device)
        for t in range(T):
            cur1 = self.pool1(self.conv1(x[:, t]))
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.pool3(self.pool2(self.conv2(spk1)))
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc1(spk2.flatten(1))
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)
            spk4, mem4 = self.lif4(cur4, mem4)

            out_spk_sum = out_spk_sum + spk4

        return out_spk_sum  # (batch, 2) rate-coded logits


def load_data(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    frames = d["frames"]  # (N, T, H, W)
    labels = d["labels"]  # (N,) strings

    x = (frames > 0).astype(np.float32)  # binarize: spike if any hit in pixel
    x = x[:, :, None, :, :]  # add channel dim -> (N, T, 1, H, W)

    label_names = sorted(set(labels.tolist()))
    label_to_idx = {name: i for i, name in enumerate(label_names)}
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    return torch.from_numpy(x), torch.from_numpy(y), label_names


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", help="frames.npz from bin_events_to_frames.py")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    x, y, label_names = load_data(args.input_npz)
    print(f"Loaded {len(x)} jets, frame shape {tuple(x.shape[1:])}, classes: {label_names}")

    n = len(x)
    idx = np.random.permutation(n)
    n_val = int(n * args.val_frac)
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    train_ds = TensorDataset(x[train_idx], y[train_idx])
    val_ds = TensorDataset(x[val_idx], y[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    n_bins = x.shape[-1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = JetSNN(n_bins=n_bins).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = loss_fn(out, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                pred = out.argmax(dim=1)
                correct += (pred == yb).sum().item()
                total += len(yb)
        val_acc = correct / total

        print(f"epoch {epoch+1:3d}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={val_acc:.3f}")


if __name__ == "__main__":
    main()
