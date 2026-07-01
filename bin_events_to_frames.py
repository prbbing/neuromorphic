"""
Bin the sparse (t, x, y) event list (from build_event_list.py) into a dense
per-jet tensor of frames: one histogram per timestep (global layer), stacked
into shape (T, H, W).

Each layer has a very different physical spread in local (x, y) -- inner
pixel layers ~+-15 cm, outer TOB layers ~+-300+ cm, growing roughly linearly
with radius since the jet cone opening angle is ~fixed. Rather than giving
every layer the same bin *count* (which makes outer-layer pixels enormous and
mostly empty/dim), we fix the physical *pixel size* (cm/bin) across all
layers and let the bin count scale with each layer's extent -- so outer
layers automatically get more, smaller (finer) pixels, and inner layers get
fewer, larger (coarser) pixels, capped by --max-bins to keep tensors a
manageable size. Every layer's histogram is then zero-padded (centered) into
a shared (max_bins, max_bins) canvas so frames still stack into one tensor.

Usage:
  python3 bin_events_to_frames.py events.h5 -o frames.npz --pixel-size 1.5
"""

import argparse

import numpy as np
import pandas as pd


def compute_layer_ranges(events, percentile=99.0):
    """Per timestep t, the symmetric (x_max, y_max) bin range sized from data."""
    ranges = {}
    lo = 100 - percentile
    for t, g in events.groupby("t"):
        x_max = np.maximum(abs(g.x.quantile(lo / 100)), abs(g.x.quantile(percentile / 100)))
        y_max = np.maximum(abs(g.y.quantile(lo / 100)), abs(g.y.quantile(percentile / 100)))
        ranges[t] = (float(x_max), float(y_max))
    return ranges


def compute_layer_bin_counts(layer_ranges, pixel_size, max_bins=64, min_bins=4):
    """For each layer, how many bins of size `pixel_size` cm are needed to
    cover its (x_max, y_max) range, capped to [min_bins, max_bins]."""
    n_bins = {}
    for t, (x_max, y_max) in layer_ranges.items():
        extent = 2 * max(x_max, y_max)
        n = int(np.ceil(extent / pixel_size))
        n_bins[t] = int(np.clip(n, min_bins, max_bins))
    return n_bins


def bin_jet_to_frames(jet_events, t_list, layer_ranges, layer_bin_counts, canvas_size):
    """Turn one jet's (t, x, y) rows into a (len(t_list), canvas_size, canvas_size)
    count tensor, with each layer's own-resolution histogram centered and
    zero-padded into the shared canvas."""
    frames = np.zeros((len(t_list), canvas_size, canvas_size), dtype=np.float32)
    for i, t in enumerate(t_list):
        sub = jet_events[jet_events.t == t]
        n_bins = layer_bin_counts[t]
        pad = (canvas_size - n_bins) // 2
        if sub.empty:
            continue
        x_max, y_max = layer_ranges[t]
        hist, _, _ = np.histogram2d(
            sub.x, sub.y, bins=n_bins,
            range=[[-x_max, x_max], [-y_max, y_max]],
        )
        frames[i, pad:pad + n_bins, pad:pad + n_bins] = hist.T  # row=y, col=x
    return frames


def build_frame_dataset(events, pixel_size=1.5, percentile=99.0, max_bins=64, min_bins=4):
    t_list = sorted(events.t.unique())
    layer_ranges = compute_layer_ranges(events, percentile=percentile)
    layer_bin_counts = compute_layer_bin_counts(layer_ranges, pixel_size, max_bins=max_bins, min_bins=min_bins)
    canvas_size = max(layer_bin_counts.values())

    jet_uids = events["jet_uid"].drop_duplicates().tolist()
    labels = events.drop_duplicates("jet_uid").set_index("jet_uid")["jet_label"]

    all_frames = np.zeros((len(jet_uids), len(t_list), canvas_size, canvas_size), dtype=np.float32)
    for i, uid in enumerate(jet_uids):
        jet_events = events[events.jet_uid == uid]
        all_frames[i] = bin_jet_to_frames(jet_events, t_list, layer_ranges, layer_bin_counts, canvas_size)

    return all_frames, labels.loc[jet_uids].to_numpy(), np.array(t_list), jet_uids, layer_bin_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", help="events.h5 produced by build_event_list.py")
    parser.add_argument("-o", "--output", default="frames.npz")
    parser.add_argument("--pixel-size", type=float, default=1.5,
                         help="physical pixel size in cm, shared across all layers (default: 1.5)")
    parser.add_argument("--max-bins", type=int, default=64,
                         help="cap on bins per axis for any one layer (default: 64)")
    parser.add_argument("--min-bins", type=int, default=4,
                         help="floor on bins per axis for any one layer (default: 4)")
    parser.add_argument("--percentile", type=float, default=99.0,
                         help="percentile used to size each layer's bin range (default: 99)")
    args = parser.parse_args()

    events = pd.read_hdf(args.input_h5, key="events")
    frames, labels, t_list, jet_uids, layer_bin_counts = build_frame_dataset(
        events, pixel_size=args.pixel_size, percentile=args.percentile,
        max_bins=args.max_bins, min_bins=args.min_bins,
    )

    print(f"bins per layer (t: n_bins): {layer_bin_counts}")
    print(f"frames shape: {frames.shape}  (n_jets, T, H, W)")
    print(f"occupancy (fraction of nonzero pixels): {(frames > 0).mean():.4f}")
    np.savez_compressed(args.output, frames=frames, labels=labels, t_list=t_list, jet_uids=jet_uids)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
