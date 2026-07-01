"""
Compute the intersection point between a jet axis and a cylindrical tracker
barrel layer, and check how well actual hits cluster around it.

The jet axis is the ray from the origin in the direction (px, py, pz). A
barrel layer is approximated as an infinite cylinder of fixed radius R
centered on the beam (z) axis. Parametrizing the axis as t*(px,py,pz)/p and
solving x^2+y^2=R^2 gives:

    x = R * cos(phi)
    y = R * sin(phi)
    z = R * sinh(eta)

i.e. only the jet direction (eta, phi) is needed, not its magnitude -- pT only
enters indirectly through how it combines with pz to set eta.

Usage:
  python3 jet_layer_intersections.py hits_jets_ttbar.h5 --outdir plots_compare
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# mean measured radius [cm] per (hit_sub_det, hit_layer), from hits_jets_ttbar.h5
LAYER_RADII = {
    (1, 1): 4.38, (1, 2): 7.29, (1, 3): 10.16,
    (3, 1): 25.52, (3, 2): 33.93, (3, 3): 41.76, (3, 4): 49.72,
    (5, 1): 60.57, (5, 2): 69.39, (5, 3): 77.96, (5, 4): 86.75, (5, 5): 96.48, (5, 6): 107.99,
}


def jet_layer_intersection(jet_eta, jet_phi, radius):
    """Return (x, y, z) where the jet axis crosses a barrel cylinder of given radius."""
    x = radius * np.cos(jet_phi)
    y = radius * np.sin(jet_phi)
    z = radius * np.sinh(jet_eta)
    return x, y, z


def add_intersections(jets, layer_radii=LAYER_RADII):
    """Add one (x,y,z) column triplet per (sub_det, layer) to a copy of jets."""
    out = jets.copy()
    for (sd, layer), radius in layer_radii.items():
        x, y, z = jet_layer_intersection(out["jet_eta"].to_numpy(), out["jet_phi"].to_numpy(), radius)
        out[f"axis_x_sd{sd}_l{layer}"] = x
        out[f"axis_y_sd{sd}_l{layer}"] = y
        out[f"axis_z_sd{sd}_l{layer}"] = z
    return out


def residual_to_nearest_hit(hits, jets, sd, layer, radius):
    """For each jet, 3D distance from the predicted axis-crossing point on this
    layer to the nearest actual hit the jet has on that layer."""
    sub_hits = hits[(hits["hit_sub_det"] == sd) & (hits["hit_layer"] == layer)]
    rows = []
    for _, jet in jets.iterrows():
        jh = sub_hits[(sub_hits["event_id"] == jet["event_id"]) & (sub_hits["jet_id"] == jet["jet_id"])]
        if jh.empty:
            continue
        ax, ay, az = jet_layer_intersection(jet["jet_eta"], jet["jet_phi"], radius)
        d = np.sqrt((jh["hit_global_x"] - ax) ** 2 + (jh["hit_global_y"] - ay) ** 2 +
                    (jh["hit_global_z"] - az) ** 2)
        rows.append(d.min())
    return np.array(rows)


def mean_residuals_per_layer(hits, jets, layer_radii=LAYER_RADII):
    sds_layers = sorted(layer_radii.keys())
    means, labels = [], []
    for (sd, layer) in sds_layers:
        d = residual_to_nearest_hit(hits, jets, sd, layer, layer_radii[(sd, layer)])
        if len(d) == 0:
            continue
        means.append(d.mean())
        labels.append(f"sd{sd}_l{layer}")
    return labels, means


def plot_residuals_per_layer(hits, jets, outdir, label="", layer_radii=LAYER_RADII):
    labels, means = mean_residuals_per_layer(hits, jets, layer_radii)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(labels, means, color="slateblue")
    ax.set_ylabel("avg. distance from predicted axis crossing\nto nearest hit [cm]")
    title = "How well actual hits cluster around the predicted jet-axis intersection"
    ax.set_title(f"{title} ({label})" if label else title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(outdir, f"axis_intersection_residuals{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_residuals_comparison(hits_list, jets_list, sample_labels, outdir, layer_radii=LAYER_RADII):
    colors = ["steelblue", "indianred", "seagreen", "darkorange"]
    width = 0.8 / len(hits_list)

    fig, ax = plt.subplots(figsize=(12, 6))
    layer_labels = None
    for i, (hits, jets, label, color) in enumerate(zip(hits_list, jets_list, sample_labels, colors)):
        layer_labels, means = mean_residuals_per_layer(hits, jets, layer_radii)
        x = np.arange(len(layer_labels)) + (i - (len(hits_list) - 1) / 2) * width
        ax.bar(x, means, width=width, color=color, label=label)

    ax.set_xticks(np.arange(len(layer_labels)))
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_ylabel("avg. distance from predicted axis crossing\nto nearest hit [cm]")
    ax.set_title("How well actual hits cluster around the predicted jet-axis intersection")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(outdir, "axis_intersection_residuals_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def wrap_angle(dphi):
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def add_local_layer_coords(hits, jets, layer_radii=LAYER_RADII):
    """For every hit (on a layer with a known radius), compute its position
    relative to its jet's predicted axis-crossing point on that layer, in a
    flattened-cylinder coordinate system:
      local_arc : signed arc length along phi (R * delta_phi)
      local_dz  : offset along z

    This recenters every jet's hits on that jet's own axis intersection, so
    hits from many different jets can be overlaid on one (local_arc, local_dz)
    plot to see a layer-by-layer "jet shape" profile.
    """
    jets_idx = jets[["event_id", "jet_id", "jet_eta", "jet_phi"]]
    merged = hits.merge(jets_idx, on=["event_id", "jet_id"], how="inner")

    radii_df = pd.DataFrame(
        [(sd, layer, r) for (sd, layer), r in layer_radii.items()],
        columns=["hit_sub_det", "hit_layer", "layer_radius"],
    )
    merged = merged.merge(radii_df, on=["hit_sub_det", "hit_layer"], how="inner")

    axis_z = merged["layer_radius"] * np.sinh(merged["jet_eta"])
    hit_phi = np.arctan2(merged["hit_global_y"], merged["hit_global_x"])
    dphi = wrap_angle(hit_phi - merged["jet_phi"])

    merged["local_arc"] = merged["layer_radius"] * dphi
    merged["local_dz"] = merged["hit_global_z"] - axis_z
    return merged


def plot_jet_shape_on_layer(hits_local, sd, layer, outdir, label=""):
    sub = hits_local[(hits_local["hit_sub_det"] == sd) & (hits_local["hit_layer"] == layer)]
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    h = ax.hist2d(sub["local_arc"], sub["local_dz"], bins=80, cmap="viridis")
    fig.colorbar(h[3], ax=ax, label="hits")
    ax.set_xlabel("local arc length along phi, R*dphi [cm]")
    ax.set_ylabel("local dz [cm]")
    title = f"Hit shape relative to jet axis, sub_det {sd} layer {layer}"
    ax.set_title(f"{title} ({label})" if label else title)
    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(outdir, f"jet_shape_sd{sd}_l{layer}{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def leading_jet(jets):
    """Return the single highest-pT jet (as a one-row DataFrame)."""
    return jets.loc[[jets["jet_pt"].idxmax()]]


def plot_leading_jet_all_layers(hits, jets, outdir, label=""):
    """For the leading jet, plot (local_arc, local_dz) hit scatter on every layer
    in a single figure with one panel per layer."""
    lead = leading_jet(jets)
    hits_local = add_local_layer_coords(hits, lead)

    layer_keys = sorted(LAYER_RADII.keys())  # (sub_det, layer) pairs in physical order
    n = len(layer_keys)
    ncols = 5
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0))

    jet_row = lead.iloc[0]
    for ax, (sd, layer) in zip(axes.flat, layer_keys):
        sub = hits_local[(hits_local["hit_sub_det"] == sd) & (hits_local["hit_layer"] == layer)]
        r = LAYER_RADII[(sd, layer)]
        ax.scatter(sub["local_arc"], sub["local_dz"], s=10, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"sd{sd} l{layer}  R={r:.1f}cm\n({len(sub)} hits)", fontsize=8)
        ax.set_xlabel("arc [cm]", fontsize=7)
        ax.set_ylabel("dz [cm]", fontsize=7)
        ax.tick_params(labelsize=6)

    # hide unused panels
    for ax in axes.flat[n:]:
        ax.set_visible(False)

    suptitle = (f"Leading jet all layers ({label})  "
                f"pT={jet_row['jet_pt']:.0f} GeV  η={jet_row['jet_eta']:.2f}")
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout()
    suffix = f"_{label}" if label else ""
    path = os.path.join(outdir, f"leading_jet_all_layers{suffix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_jet_all_layers(hits, jet_row, outdir, tag=""):
    """Plot (local_arc, local_dz) hit scatter across all layers for a single jet row."""
    single_jet = pd.DataFrame([jet_row])
    hits_local = add_local_layer_coords(hits, single_jet)

    layer_keys = sorted(LAYER_RADII.keys())
    ncols = 5
    nrows = (len(layer_keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0))

    for ax, (sd, layer) in zip(axes.flat, layer_keys):
        sub = hits_local[(hits_local["hit_sub_det"] == sd) & (hits_local["hit_layer"] == layer)]
        r = LAYER_RADII[(sd, layer)]
        ax.scatter(sub["local_arc"], sub["local_dz"], s=10, alpha=0.7)
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax.set_title(f"sd{sd} l{layer}  R={r:.1f}cm\n({len(sub)} hits)", fontsize=8)
        ax.set_xlabel("arc [cm]", fontsize=7)
        ax.set_ylabel("dz [cm]", fontsize=7)
        ax.tick_params(labelsize=6)

    for ax in axes.flat[len(layer_keys):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Jet {tag}  pT={jet_row['jet_pt']:.0f} GeV  η={jet_row['jet_eta']:.2f}",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, f"jet_all_layers_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_leading_jet_all_layers(hits, jets, outdir, label=""):
    lead = leading_jet(jets).iloc[0]
    tag = f"leading_{label}" if label else "leading"
    plot_jet_all_layers(hits, lead, outdir, tag=tag)


def plot_random_jets_all_layers(hits, jets, outdir, n=10, seed=None, label=""):
    """Plot all-layer hit patterns for n randomly sampled jets."""
    sample = jets.sample(min(n, len(jets)), random_state=seed)
    for _, jet_row in sample.iterrows():
        tag = f"{label}_ev{jet_row['event_id']}_j{jet_row['jet_id']}" if label else f"ev{jet_row['event_id']}_j{jet_row['jet_id']}"
        plot_jet_all_layers(hits, jet_row, outdir, tag=tag)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", nargs="+", help="One or more hits_jets*.h5 files to process/compare")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Labels for each input file, same order/count as input_h5 "
                              "(default: derived from filenames)")
    parser.add_argument("--outdir", default="plots_intersections")
    parser.add_argument("--max-jets", type=int, default=300,
                         help="Limit number of jets used per file for the residual scan "
                              "(it's O(n_jets * n_layers))")
    parser.add_argument("--random-jets", type=int, default=0,
                         help="Plot all-layer hit patterns for this many randomly sampled jets "
                              "per file (default: 0, disabled)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Random seed for --random-jets sampling")
    parser.add_argument("--shape-layers", nargs="*", type=int, default=[1, 1, 5, 6],
                         metavar="SD LAYER", help="Pairs of (hit_sub_det, hit_layer) to make "
                              "jet-shape plots for, e.g. --shape-layers 1 1 5 6 for "
                              "PixelBarrel layer 1 and TOB layer 6 (default: 1 1 5 6)")
    args = parser.parse_args()

    if len(args.shape_layers) % 2 != 0:
        parser.error("--shape-layers must be given as (sub_det, layer) pairs")
    shape_layers = list(zip(args.shape_layers[0::2], args.shape_layers[1::2]))

    labels = args.labels or [os.path.splitext(os.path.basename(p))[0] for p in args.input_h5]
    if len(labels) != len(args.input_h5):
        parser.error("--labels must have the same number of entries as input_h5 files")

    os.makedirs(args.outdir, exist_ok=True)

    hits_list, jets_list = [], []
    for path, label in zip(args.input_h5, labels):
        hits = pd.read_hdf(path, key="hits")
        jets = pd.read_hdf(path, key="jets")
        if args.max_jets is not None and len(jets) > args.max_jets:
            jets = jets.sample(args.max_jets, random_state=0)
        hits_list.append(hits)
        jets_list.append(jets)
        print(f"{label}: loaded {len(hits)} hits, using {len(jets)} jets for residual scan")
        plot_residuals_per_layer(hits, jets, args.outdir, label=label)

        hits_local = add_local_layer_coords(hits, jets)
        for sd, layer in shape_layers:
            plot_jet_shape_on_layer(hits_local, sd, layer, args.outdir, label=label)

        plot_leading_jet_all_layers(hits, jets, args.outdir, label=label)
        if args.random_jets > 0:
            plot_random_jets_all_layers(hits, jets, args.outdir, n=args.random_jets,
                                        seed=args.seed, label=label)

    if len(args.input_h5) > 1:
        plot_residuals_comparison(hits_list, jets_list, labels, args.outdir)

    print("Done.")


if __name__ == "__main__":
    main()
