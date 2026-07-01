"""
Quick analysis/plotting script for the hit-jet association tables produced by
extract_hit_jet_association.py (hits_jets.h5: "hits" and "jets" keys).

Produces a set of diagnostic plots:
  - jet kinematics: pt, eta, phi
  - n_hits per jet distribution
  - hits per detector layer (split by sub-detector)
  - hits per global detector layer (continuous numbering across sub-detectors)
  - average within-jet pairwise hit distance per layer (split by sub-detector)
  - hit global x/y/z distributions

Plots are saved as PNG files into an output directory.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def savefig(fig, outdir, name):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_jet_kinematics(jets, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    axes[0].hist(jets["jet_pt"], bins=60, color="steelblue")
    axes[0].set_xlabel("jet pT [GeV]")
    axes[0].set_ylabel("jets")
    axes[0].set_yscale("log")

    axes[1].hist(jets["jet_eta"], bins=60, color="indianred")
    axes[1].set_xlabel("jet eta")
    axes[1].set_ylabel("jets")

    axes[2].hist(jets["jet_phi"], bins=60, color="darkorange")
    axes[2].set_xlabel("jet phi [rad]")
    axes[2].set_ylabel("jets")

    fig.suptitle("Jet kinematics")
    fig.tight_layout()
    savefig(fig, outdir, "jet_kinematics")


def plot_n_hits_per_jet(jets, outdir):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(jets["n_hits"], bins=60, color="purple")
    ax.set_xlabel("n_hits per jet")
    ax.set_ylabel("jets")
    ax.set_yscale("log")
    ax.set_title("Number of matched hits per jet")
    fig.tight_layout()
    savefig(fig, outdir, "n_hits_per_jet")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(jets["jet_pt"], jets["n_hits"], s=4, alpha=0.3, color="purple")
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("n_hits")
    ax.set_yscale("log")
    ax.set_title("n_hits vs jet pT")
    fig.tight_layout()
    savefig(fig, outdir, "n_hits_vs_jet_pt")


# Standard CMS tracker layer/disk counts per sub-detector, ordered roughly
# inner-to-outer in radius: PixelBarrel(1), PixelEndcap(2), TIB(3), TID(4), TOB(5), TEC(6)
SUB_DET_NAMES = {1: "PixelBarrel", 2: "PixelEndcap", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
SUB_DET_N_LAYERS = {1: 3, 2: 2, 3: 4, 4: 3, 5: 6, 6: 9}
SUB_DET_ORDER = [1, 2, 3, 4, 5, 6]


def global_layer_offsets():
    offsets = {}
    running = 0
    for sd in SUB_DET_ORDER:
        offsets[sd] = running
        running += SUB_DET_N_LAYERS[sd]
    return offsets


def add_global_layer(hits):
    offsets = global_layer_offsets()
    offset_arr = hits["hit_sub_det"].map(offsets)
    return hits["hit_layer"] + offset_arr


def plot_hits_per_global_layer(hits, outdir, n_jets):
    global_layer = add_global_layer(hits)
    counts = global_layer.value_counts().sort_index()
    avg_counts = counts / n_jets

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(avg_counts.index, avg_counts.values, color="steelblue", width=0.8)
    ax.set_xlabel("global detector layer")
    ax.set_ylabel("avg. number of hits per jet")
    ax.set_title("Average hits per jet, per global detector layer")

    offsets = global_layer_offsets()
    present_sub_dets = sorted(hits["hit_sub_det"].unique())
    for sd in present_sub_dets:
        boundary = offsets[sd] + 0.5
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
        ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd], ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, outdir, "avg_hits_per_jet_per_global_layer")


def plot_hits_per_layer(hits, outdir, n_jets):
    sub_dets = sorted(hits["hit_sub_det"].unique())
    fig, ax = plt.subplots(figsize=(9, 6))

    for sd in sub_dets:
        sub = hits[hits["hit_sub_det"] == sd]
        counts = sub["hit_layer"].value_counts().sort_index()
        ax.plot(counts.index, counts.values, marker="o", label=f"sub_det {sd}")

    ax.set_xlabel("layer")
    ax.set_ylabel("number of hits")
    ax.set_yscale("log")
    ax.set_title("Hits per layer, split by sub-detector")
    ax.legend()
    fig.tight_layout()
    savefig(fig, outdir, "hits_per_layer")

    fig, ax = plt.subplots(figsize=(7, 5))
    sub_counts = hits["hit_sub_det"].value_counts().sort_index()
    ax.bar(sub_counts.index.astype(str), sub_counts.values, color="teal")
    ax.set_xlabel("sub_det")
    ax.set_ylabel("number of hits")
    ax.set_title("Hits per sub-detector")
    fig.tight_layout()
    savefig(fig, outdir, "hits_per_subdet")

    fig, ax = plt.subplots(figsize=(9, 6))
    for sd in sub_dets:
        sub = hits[hits["hit_sub_det"] == sd]
        avg_counts = sub["hit_layer"].value_counts().sort_index() / n_jets
        ax.plot(avg_counts.index, avg_counts.values, marker="o", label=f"sub_det {sd}")

    ax.set_xlabel("layer")
    ax.set_ylabel("avg. number of hits per jet")
    ax.set_title("Average hits per jet, per layer, split by sub-detector")
    ax.legend()
    fig.tight_layout()
    savefig(fig, outdir, "avg_hits_per_jet_per_layer")


def mean_nearest_neighbor_distance(group):
    coords = group[["hit_global_x", "hit_global_y", "hit_global_z"]].drop_duplicates().to_numpy()
    if len(coords) < 2:
        return np.nan
    tree = cKDTree(coords)
    # k=2 because the nearest neighbor to a point in its own tree is itself (distance 0)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1].mean()


def plot_avg_hit_distance_per_layer(hits, outdir):
    # exact-duplicate hit records (same event/jet/layer/position) are a known artifact
    # in the strip tracker layers (TIB layers 3-4, TOB layers 3-6 show ~50% duplication)
    # and would otherwise crush the nearest-neighbor distance toward zero; dedupe first.
    per_jet_layer = (
        hits.groupby(["event_id", "jet_id", "hit_sub_det", "hit_layer"])
        .apply(mean_nearest_neighbor_distance, include_groups=False)
        .rename("mean_nn_dist")
        .reset_index()
        .dropna(subset=["mean_nn_dist"])
    )

    avg_per_layer = (
        per_jet_layer.groupby(["hit_sub_det", "hit_layer"])["mean_nn_dist"]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(9, 6))
    for sd in sorted(avg_per_layer["hit_sub_det"].unique()):
        sub = avg_per_layer[avg_per_layer["hit_sub_det"] == sd].sort_values("hit_layer")
        ax.plot(sub["hit_layer"], sub["mean_nn_dist"], marker="o", label=f"sub_det {sd}")

    ax.set_xlabel("layer")
    ax.set_ylabel("avg. nearest-neighbor hit distance within jet [cm]")
    ax.set_title("Average within-jet nearest-neighbor hit distance per layer, split by sub-detector")
    ax.legend()
    fig.tight_layout()
    savefig(fig, outdir, "avg_hit_distance_per_layer")

    offsets = global_layer_offsets()
    avg_per_layer["global_layer"] = avg_per_layer["hit_layer"] + avg_per_layer["hit_sub_det"].map(offsets)
    avg_per_layer = avg_per_layer.sort_values("global_layer")

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(avg_per_layer["global_layer"], avg_per_layer["mean_nn_dist"], color="darkorange", width=0.8)
    ax.set_xlabel("global detector layer")
    ax.set_ylabel("avg. nearest-neighbor hit distance within jet [cm]")
    ax.set_title("Average within-jet nearest-neighbor hit distance, per global detector layer")

    present_sub_dets = sorted(hits["hit_sub_det"].unique())
    for sd in present_sub_dets:
        boundary = offsets[sd] + 0.5
        ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
        mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
        ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd], ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    savefig(fig, outdir, "avg_hit_distance_per_global_layer")


def plot_hit_positions(hits, outdir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].hist(hits["hit_global_x"], bins=100, color="gray")
    axes[0].set_xlabel("hit global x [cm]")
    axes[1].hist(hits["hit_global_y"], bins=100, color="gray")
    axes[1].set_xlabel("hit global y [cm]")
    axes[2].hist(hits["hit_global_z"], bins=100, color="gray")
    axes[2].set_xlabel("hit global z [cm]")
    for a in axes:
        a.set_ylabel("hits")
        a.set_yscale("log")
    fig.suptitle("Hit global position distributions")
    fig.tight_layout()
    savefig(fig, outdir, "hit_global_positions")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(hits["hit_global_x"], hits["hit_global_y"], s=0.5, alpha=0.1, color="black")
    ax.set_xlabel("hit global x [cm]")
    ax.set_ylabel("hit global y [cm]")
    ax.set_title("Hit positions, transverse (x-y) view")
    ax.set_aspect("equal")
    fig.tight_layout()
    savefig(fig, outdir, "hit_xy_view")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", help="Path to hits_jets.h5 produced by extract_hit_jet_association.py")
    parser.add_argument("--outdir", default="plots", help="Directory to write plots into")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hits = pd.read_hdf(args.input_h5, key="hits")
    jets = pd.read_hdf(args.input_h5, key="jets")

    print(f"Loaded {len(hits)} hits and {len(jets)} jets")

    print("Plotting jet kinematics...")
    plot_jet_kinematics(jets, args.outdir)
    print("Plotting n_hits per jet...")
    plot_n_hits_per_jet(jets, args.outdir)
    n_jets = len(jets)
    print("Plotting hits per layer...")
    plot_hits_per_layer(hits, args.outdir, n_jets)
    print("Plotting hits per global layer...")
    plot_hits_per_global_layer(hits, args.outdir, n_jets)
    print("Plotting average within-jet hit distance per layer...")
    plot_avg_hit_distance_per_layer(hits, args.outdir)
    print("Plotting hit positions...")
    plot_hit_positions(hits, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
