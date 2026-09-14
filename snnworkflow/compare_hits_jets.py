"""
Compare two hit-jet association tables (e.g. QCD vs ttbar) produced by
extract_hit_jet_association.py. Produces overlaid diagnostic plots so the
two samples can be visually compared.

Usage:
  python3 compare_hits_jets.py hits_jets_qcd.h5 hits_jets_ttbar.h5 \
      --labels QCD ttbar --outdir plots_compare
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

SUB_DET_NAMES = {1: "PixelBarrel", 2: "PixelEndcap", 3: "TIB", 4: "TID", 5: "TOB", 6: "TEC"}
SUB_DET_N_LAYERS = {1: 3, 2: 2, 3: 4, 4: 3, 5: 6, 6: 9}
SUB_DET_ORDER = [1, 2, 3, 4, 5, 6]
COLORS = ["steelblue", "indianred", "seagreen", "darkorange"]


def savefig(fig, outdir, name):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


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


def leading_jets(jets):
    """Return only the highest-pT jet per event."""
    return jets.loc[jets.groupby("event_id")["jet_pt"].idxmax()]


def plot_jet_kinematics(jets_list, labels, outdir):
    for suffix, jets_sel in [("all", lambda j: j), ("leading", leading_jets)]:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        fields = [("jet_pt", "jet pT [GeV]", True),
                  ("jet_eta", "jet eta", False), ("jet_phi", "jet phi [rad]", False)]
        for ax, (field, xlabel, logy) in zip(axes.flat, fields):
            for jets, label, color in zip(jets_list, labels, COLORS):
                sel = jets_sel(jets)
                ax.hist(sel[field], bins=60, histtype="step", density=True,
                        label=label, color=color, linewidth=1.5)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("jets (normalized)")
            if logy:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
        title = "Jet kinematics comparison" + (" (leading jet only)" if suffix == "leading" else "")
        fig.suptitle(title)
        fig.tight_layout()
        savefig(fig, outdir, f"compare_jet_kinematics_{suffix}")


def plot_n_hits_per_jet(jets_list, labels, outdir):
    for suffix, jets_sel in [("all", lambda j: j), ("leading", leading_jets)]:
        stitle = " (leading jet only)" if suffix == "leading" else ""

        fig, ax = plt.subplots(figsize=(7, 5))
        for jets, label, color in zip(jets_list, labels, COLORS):
            sel = jets_sel(jets)
            ax.hist(sel["n_hits"], bins=60, histtype="step", density=True,
                    label=label, color=color, linewidth=1.5)
        ax.set_xlabel("n_hits per jet")
        ax.set_ylabel("jets (normalized)")
        ax.set_yscale("log")
        ax.set_title(f"Number of matched hits per jet{stitle}")
        ax.legend()
        fig.tight_layout()
        savefig(fig, outdir, f"compare_n_hits_per_jet_{suffix}")

        fig, ax = plt.subplots(figsize=(7, 5))
        for jets, label, color in zip(jets_list, labels, COLORS):
            sel = jets_sel(jets)
            ax.scatter(sel["jet_pt"], sel["n_hits"], s=4, alpha=0.25, color=color, label=label)
        ax.set_xlabel("jet pT [GeV]")
        ax.set_ylabel("n_hits")
        ax.set_yscale("log")
        ax.set_title(f"n_hits vs jet pT{stitle}")
        ax.legend()
        fig.tight_layout()
        savefig(fig, outdir, f"compare_n_hits_vs_jet_pt_{suffix}")


def leading_hits(hits, jets):
    """Return only hits belonging to the leading jet (highest pT) in each event."""
    lead = leading_jets(jets)[["event_id", "jet_id"]]
    return hits.merge(lead, on=["event_id", "jet_id"], how="inner")


def plot_hits_per_global_layer(hits_list, labels, n_jets_list, outdir, jets_list=None):
    offsets = global_layer_offsets()

    selections = [("all", hits_list, n_jets_list)]
    if jets_list is not None:
        lead_hits_list = [leading_hits(h, j) for h, j in zip(hits_list, jets_list)]
        lead_n_jets = [leading_jets(j).shape[0] for j in jets_list]
        selections.append(("leading", lead_hits_list, lead_n_jets))

    for suffix, sel_hits_list, sel_n_jets_list in selections:
        fig, ax = plt.subplots(figsize=(11, 6))
        width = 0.8 / len(sel_hits_list)

        for i, (hits, label, n_jets, color) in enumerate(zip(sel_hits_list, labels, sel_n_jets_list, COLORS)):
            global_layer = add_global_layer(hits)
            counts = global_layer.value_counts().sort_index()
            avg_counts = counts / n_jets
            offset = (i - (len(sel_hits_list) - 1) / 2) * width
            ax.bar(avg_counts.index + offset, avg_counts.values, width=width, color=color, label=label)

        stitle = " (leading jet only)" if suffix == "leading" else ""
        ax.set_xlabel("global detector layer")
        ax.set_ylabel("avg. number of hits per jet")
        ax.set_title(f"Average hits per jet, per global detector layer{stitle}")

        present_sub_dets = sorted(set().union(*[h["hit_sub_det"].unique() for h in sel_hits_list]))
        for sd in present_sub_dets:
            boundary = offsets[sd] + 0.5
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
            mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
            ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd], ha="center", va="bottom", fontsize=9)
        ax.legend()
        fig.tight_layout()
        savefig(fig, outdir, f"compare_avg_hits_per_jet_per_global_layer_{suffix}")


def mean_nearest_neighbor_distance(group):
    coords = group[["hit_global_x", "hit_global_y", "hit_global_z"]].drop_duplicates().to_numpy()
    if len(coords) < 2:
        return np.nan
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1].mean()


def plot_avg_hit_distance_per_global_layer(hits_list, labels, outdir, jets_list=None):
    offsets = global_layer_offsets()

    selections = [("all", hits_list)]
    if jets_list is not None:
        lead_hits_list = [leading_hits(h, j) for h, j in zip(hits_list, jets_list)]
        selections.append(("leading", lead_hits_list))

    for suffix, sel_hits_list in selections:
        fig, ax = plt.subplots(figsize=(11, 6))
        width = 0.8 / len(sel_hits_list)
        all_sub_dets = set()

        for i, (hits, label, color) in enumerate(zip(sel_hits_list, labels, COLORS)):
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
            avg_per_layer["global_layer"] = avg_per_layer["hit_layer"] + avg_per_layer["hit_sub_det"].map(offsets)
            avg_per_layer = avg_per_layer.sort_values("global_layer")
            offset = (i - (len(sel_hits_list) - 1) / 2) * width
            ax.bar(avg_per_layer["global_layer"] + offset, avg_per_layer["mean_nn_dist"],
                   width=width, color=color, label=label)
            all_sub_dets |= set(hits["hit_sub_det"].unique())

        stitle = " (leading jet only)" if suffix == "leading" else ""
        ax.set_xlabel("global detector layer")
        ax.set_ylabel("avg. nearest-neighbor hit distance within jet [cm]")
        ax.set_title(f"Average within-jet nearest-neighbor hit distance, per global detector layer{stitle}")

        for sd in sorted(all_sub_dets):
            boundary = offsets[sd] + 0.5
            ax.axvline(boundary, color="gray", linestyle="--", linewidth=0.8)
            mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
            ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd], ha="center", va="bottom", fontsize=9)
        ax.legend()
        fig.tight_layout()
        savefig(fig, outdir, f"compare_avg_hit_distance_per_global_layer_{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", nargs="+", help="Two or more .h5 files to compare")
    parser.add_argument("--labels", nargs="+", required=True,
                         help="Labels for each input file, same order/count as input_h5")
    parser.add_argument("--outdir", default="plots_compare", help="Directory to write plots into")
    args = parser.parse_args()

    if len(args.input_h5) != len(args.labels):
        parser.error("--labels must have the same number of entries as input_h5 files")

    os.makedirs(args.outdir, exist_ok=True)

    hits_list, jets_list = [], []
    for path in args.input_h5:
        hits_list.append(pd.read_hdf(path, key="hits"))
        jets_list.append(pd.read_hdf(path, key="jets"))

    for label, hits, jets in zip(args.labels, hits_list, jets_list):
        print(f"{label}: {len(hits)} hits, {len(jets)} jets")

    print("Plotting jet kinematics comparison...")
    plot_jet_kinematics(jets_list, args.labels, args.outdir)
    print("Plotting n_hits per jet comparison...")
    plot_n_hits_per_jet(jets_list, args.labels, args.outdir)
    print("Plotting hits per global layer comparison...")
    n_jets_list = [len(j) for j in jets_list]
    plot_hits_per_global_layer(hits_list, args.labels, n_jets_list, args.outdir, jets_list=jets_list)
    print("Plotting average within-jet hit distance comparison...")
    plot_avg_hit_distance_per_global_layer(hits_list, args.labels, args.outdir, jets_list=jets_list)

    print("Done.")


if __name__ == "__main__":
    main()
