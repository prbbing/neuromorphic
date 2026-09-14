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


FLAVOUR_STYLE = {
    "b":     {"color": "#2a78d6", "label": "b-jets"},
    "c":     {"color": "#eda100", "label": "c-jets"},
    "light": {"color": "#1baf7a", "label": "light-jets"},
}
FLAVOURS = list(FLAVOUR_STYLE.keys())


def merge_flavour(hits, jets):
    """Return hits with a 'jet_label' column merged in from jets."""
    return hits.merge(
        jets[["event_id", "jet_id", "jet_label"]],
        on=["event_id", "jet_id"], how="left",
    )


_FALLBACK_COLORS = ["#2a78d6", "#eda100", "#1baf7a", "#e34948", "#888780"]


def flavour_subsets(df, label_col="jet_label"):
    """Yield (i, flavour, subset_df) for each label present in df, in a consistent order."""
    present = sorted(df[label_col].dropna().unique())
    for i, fl in enumerate(present):
        sub = df[df[label_col] == fl]
        if len(sub):
            yield i, fl, sub


def flavour_style(fl, i=0):
    """Return (color, label) for a flavour string, falling back gracefully."""
    if fl in FLAVOUR_STYLE:
        return FLAVOUR_STYLE[fl]["color"], FLAVOUR_STYLE[fl]["label"]
    return _FALLBACK_COLORS[i % len(_FALLBACK_COLORS)], fl


def savefig(fig, outdir, name):
    path = os.path.join(outdir, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def plot_jet_kinematics(jets, outdir, by_flavour=False):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    vars_ = [
        ("jet_pt",  "jet pT [GeV]",  True),
        ("jet_eta", "jet eta",        False),
        ("jet_phi", "jet phi [rad]",  False),
    ]
    if by_flavour:
        for ax, (col, xlabel, log) in zip(axes, vars_):
            for i, fl, sub in flavour_subsets(jets):
                ax.hist(sub[col], bins=60, alpha=0.6,
                        color=flavour_style(fl, i)[0],
                        label=flavour_style(fl, i)[1],
                        density=True)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            if log:
                ax.set_yscale("log")
            ax.legend(fontsize=8)
    else:
        colors = ["steelblue", "indianred", "darkorange"]
        for ax, (col, xlabel, log), color in zip(axes, vars_, colors):
            ax.hist(jets[col], bins=60, color=color)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("jets")
            if log:
                ax.set_yscale("log")

    fig.suptitle("Jet kinematics" + (" by flavour" if by_flavour else ""))
    fig.tight_layout()
    savefig(fig, outdir, "jet_kinematics" + ("_by_flavour" if by_flavour else ""))


def plot_n_hits_per_jet(jets, outdir, by_flavour=False):
    fig, ax = plt.subplots(figsize=(7, 5))
    if by_flavour:
        for i, fl, sub in flavour_subsets(jets):
            ax.hist(sub["n_hits"], bins=60, alpha=0.6, density=True,
                    color=FLAVOUR_STYLE[fl]["color"], label=FLAVOUR_STYLE[fl]["label"])
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
    else:
        ax.hist(jets["n_hits"], bins=60, color="purple")
        ax.set_ylabel("jets")
    ax.set_xlabel("n_hits per jet")
    ax.set_yscale("log")
    ax.set_title("Number of matched hits per jet" + (" by flavour" if by_flavour else ""))
    fig.tight_layout()
    savefig(fig, outdir, "n_hits_per_jet" + ("_by_flavour" if by_flavour else ""))

    fig, ax = plt.subplots(figsize=(7, 5))
    if by_flavour:
        for i, fl, sub in flavour_subsets(jets):
            ax.scatter(sub["jet_pt"], sub["n_hits"], s=4, alpha=0.2,
                       color=FLAVOUR_STYLE[fl]["color"], label=FLAVOUR_STYLE[fl]["label"])
        ax.legend(fontsize=8)
    else:
        ax.scatter(jets["jet_pt"], jets["n_hits"], s=4, alpha=0.3, color="purple")
    ax.set_xlabel("jet pT [GeV]")
    ax.set_ylabel("n_hits")
    ax.set_yscale("log")
    ax.set_title("n_hits vs jet pT" + (" by flavour" if by_flavour else ""))
    fig.tight_layout()
    savefig(fig, outdir, "n_hits_vs_jet_pt" + ("_by_flavour" if by_flavour else ""))


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


def plot_hits_per_global_layer(hits, outdir, n_jets, by_flavour=False):
    offsets = global_layer_offsets()
    present_sub_dets = sorted(hits["hit_sub_det"].unique())

    def _add_subdet_labels(ax):
        for sd in present_sub_dets:
            ax.axvline(offsets[sd] + 0.5, color="gray", linestyle="--", linewidth=0.8)
            mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
            ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd],
                    ha="center", va="bottom", fontsize=9)

    fig, ax = plt.subplots(figsize=(11, 6))
    if by_flavour:
        hits_fl = hits.copy()
        hits_fl["global_layer"] = add_global_layer(hits_fl)
        all_layers = sorted(hits_fl["global_layer"].unique())
        present_fl = sorted(hits_fl["jet_label"].dropna().unique())
        width = 0.8 / max(len(present_fl), 1)
        for i, fl, sub in flavour_subsets(hits_fl, label_col="jet_label"):
            n_fl = sub.groupby(["event_id", "jet_id"]).ngroups
            counts = sub["global_layer"].value_counts().reindex(all_layers, fill_value=0)
            avg = counts / max(n_fl, 1)
            offset = (i - (len(present_fl) - 1) / 2) * width
            color, label = flavour_style(fl, i)
            ax.bar(np.array(all_layers, dtype=float) + offset,
                   avg.values, width=width, alpha=0.8, color=color, label=label)
        ax.legend(fontsize=8)
    else:
        global_layer = add_global_layer(hits)
        avg_counts = global_layer.value_counts().sort_index() / n_jets
        ax.bar(avg_counts.index, avg_counts.values, color="steelblue", width=0.8)

    ax.set_xlabel("global detector layer")
    ax.set_ylabel("avg. number of hits per jet")
    ax.set_title("Average hits per jet, per global detector layer"
                 + (" by flavour" if by_flavour else ""))
    _add_subdet_labels(ax)
    fig.tight_layout()
    savefig(fig, outdir, "avg_hits_per_jet_per_global_layer"
            + ("_by_flavour" if by_flavour else ""))


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


def plot_avg_hit_distance_per_layer(hits, outdir, by_flavour=False):
    # exact-duplicate hit records (same event/jet/layer/position) are a known artifact
    # in the strip tracker layers (TIB layers 3-4, TOB layers 3-6 show ~50% duplication)
    # and would otherwise crush the nearest-neighbor distance toward zero; dedupe first.
    groupby_cols = ["event_id", "jet_id", "hit_sub_det", "hit_layer"]
    if by_flavour and "jet_label" in hits.columns:
        groupby_cols = ["jet_label"] + groupby_cols

    per_jet_layer = (
        hits.groupby(groupby_cols)
        .apply(mean_nearest_neighbor_distance, include_groups=False)
        .rename("mean_nn_dist")
        .reset_index()
        .dropna(subset=["mean_nn_dist"])
    )

    offsets = global_layer_offsets()

    # --- per sub-detector line plot ---
    fig, ax = plt.subplots(figsize=(9, 6))
    if by_flavour and "jet_label" in per_jet_layer.columns:
        for i, fl, sub in flavour_subsets(per_jet_layer):
            avg = sub.groupby(["hit_sub_det", "hit_layer"])["mean_nn_dist"].mean().reset_index()
            for sd in sorted(avg["hit_sub_det"].unique()):
                sd_sub = avg[avg["hit_sub_det"] == sd].sort_values("hit_layer")
                color, label = flavour_style(fl, i)
                ax.plot(sd_sub["hit_layer"] + offsets[sd], sd_sub["mean_nn_dist"],
                        marker="o", color=color,
                        label=f"{label} sd{sd}", linestyle="-" if i == 0 else "--")
    else:
        avg_per_layer = (
            per_jet_layer.groupby(["hit_sub_det", "hit_layer"])["mean_nn_dist"]
            .mean().reset_index()
        )
        for sd in sorted(avg_per_layer["hit_sub_det"].unique()):
            sub = avg_per_layer[avg_per_layer["hit_sub_det"] == sd].sort_values("hit_layer")
            ax.plot(sub["hit_layer"], sub["mean_nn_dist"], marker="o", label=f"sub_det {sd}")

    ax.set_xlabel("layer")
    ax.set_ylabel("avg. nearest-neighbor hit distance within jet [cm]")
    ax.set_title("Average within-jet nearest-neighbor hit distance per layer"
                 + (" by flavour" if by_flavour else ""))
    ax.legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, outdir, "avg_hit_distance_per_layer"
            + ("_by_flavour" if by_flavour else ""))

    # --- global layer bar chart ---
    fig, ax = plt.subplots(figsize=(11, 6))
    present_sub_dets = sorted(hits["hit_sub_det"].unique())

    if by_flavour and "jet_label" in per_jet_layer.columns:
        flavours_present = sorted(per_jet_layer["jet_label"].dropna().unique())
        n_fl = len(flavours_present)
        width = 0.8 / max(n_fl, 1)
        for i, fl, sub in flavour_subsets(per_jet_layer):
            avg = sub.groupby(["hit_sub_det", "hit_layer"])["mean_nn_dist"].mean().reset_index()
            avg["global_layer"] = avg["hit_layer"] + avg["hit_sub_det"].map(offsets)
            avg = avg.sort_values("global_layer")
            offset = (i - (n_fl - 1) / 2) * width
            color, label = flavour_style(fl, i)
            ax.bar(avg["global_layer"] + offset, avg["mean_nn_dist"],
                   width=width, alpha=0.8, color=color, label=label)
        ax.legend(fontsize=8)
    else:
        avg_per_layer = (
            per_jet_layer.groupby(["hit_sub_det", "hit_layer"])["mean_nn_dist"]
            .mean().reset_index()
        )
        avg_per_layer["global_layer"] = (avg_per_layer["hit_layer"]
                                         + avg_per_layer["hit_sub_det"].map(offsets))
        avg_per_layer = avg_per_layer.sort_values("global_layer")
        ax.bar(avg_per_layer["global_layer"], avg_per_layer["mean_nn_dist"],
               color="darkorange", width=0.8)

    ax.set_xlabel("global detector layer")
    ax.set_ylabel("avg. nearest-neighbor hit distance within jet [cm]")
    ax.set_title("Average within-jet nearest-neighbor hit distance, per global detector layer"
                 + (" by flavour" if by_flavour else ""))
    for sd in present_sub_dets:
        ax.axvline(offsets[sd] + 0.5, color="gray", linestyle="--", linewidth=0.8)
        mid = offsets[sd] + SUB_DET_N_LAYERS[sd] / 2 + 0.5
        ax.text(mid, ax.get_ylim()[1], SUB_DET_NAMES[sd],
                ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    savefig(fig, outdir, "avg_hit_distance_per_global_layer"
            + ("_by_flavour" if by_flavour else ""))


def plot_hit_positions(hits, outdir, by_flavour=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    coords = [("hit_global_x", "hit global x [cm]"),
              ("hit_global_y", "hit global y [cm]"),
              ("hit_global_z", "hit global z [cm]")]
    if by_flavour:
        for ax, (col, xlabel) in zip(axes, coords):
            for i, fl, sub in flavour_subsets(hits, label_col="jet_label"):
                ax.hist(sub[col], bins=100, alpha=0.5, density=True,
                        color=FLAVOUR_STYLE[fl]["color"], label=FLAVOUR_STYLE[fl]["label"])
            ax.set_xlabel(xlabel)
            ax.set_ylabel("density")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
    else:
        for ax, (col, xlabel) in zip(axes, coords):
            ax.hist(hits[col], bins=100, color="gray")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("hits")
            ax.set_yscale("log")
    fig.suptitle("Hit global position distributions" + (" by flavour" if by_flavour else ""))
    fig.tight_layout()
    savefig(fig, outdir, "hit_global_positions" + ("_by_flavour" if by_flavour else ""))

    fig, ax = plt.subplots(figsize=(7, 7))
    if by_flavour:
        for i, fl, sub in flavour_subsets(hits, label_col="jet_label"):
            sample = sub.sample(min(len(sub), 20_000), random_state=42)
            ax.scatter(sample["hit_global_x"], sample["hit_global_y"],
                       s=0.5, alpha=0.15,
                       color=FLAVOUR_STYLE[fl]["color"], label=FLAVOUR_STYLE[fl]["label"])
        ax.legend(fontsize=8, markerscale=6)
    else:
        ax.scatter(hits["hit_global_x"], hits["hit_global_y"], s=0.5, alpha=0.1, color="black")
    ax.set_xlabel("hit global x [cm]")
    ax.set_ylabel("hit global y [cm]")
    ax.set_title("Hit positions, transverse (x-y) view" + (" by flavour" if by_flavour else ""))
    ax.set_aspect("equal")
    fig.tight_layout()
    savefig(fig, outdir, "hit_xy_view" + ("_by_flavour" if by_flavour else ""))


def plot_single_jets(hits, jets, outdir, n_jets=5, seed=42, local=True):
    """
    For each flavour, sample n_jets random jets and plot their hits in
    global (x,y), (x,z), (y,z) and — if local=True — local (x',y'), (x',z'), (y',z').
    One PNG per jet: <outdir>/single_jets/<flavour>_<event>_<jet>.png
    """
    from jet_layer_intersections import add_local_layer_coords, LAYER_RADII

    rng = np.random.default_rng(seed)
    out_base = os.path.join(outdir, "single_jets")
    os.makedirs(out_base, exist_ok=True)

    if "jet_label" not in jets.columns:
        print("  [skip] single_jets: no jet_label column — run with --truth-flavor")
        return

    if local:
        hits_local = add_local_layer_coords(hits, jets,
                                            layer_radii=LAYER_RADII,
                                            use_jet_perp=True)
        # local_arc = x', local_dz = y'; compute z' separately
        jets_idx = jets[["event_id", "jet_id", "jet_eta", "jet_phi"]]
        hits_local = hits_local.merge(jets_idx, on=["event_id", "jet_id"],
                                      how="left", suffixes=("", "_j"))
        radii_df = pd.DataFrame(
            [(sd, ly, r) for (sd, ly), r in LAYER_RADII.items()],
            columns=["hit_sub_det", "hit_layer", "layer_radius"],
        )
        hits_local = hits_local.merge(radii_df, on=["hit_sub_det", "hit_layer"],
                                      how="left", suffixes=("", "_r"))
        eta_col = "jet_eta" if "jet_eta" in hits_local.columns else "jet_eta_j"
        hits_local["local_z"] = (hits_local["hit_global_z"]
                                 - hits_local["layer_radius"] * np.sinh(hits_local[eta_col]))

    for i, fl, fl_jets in flavour_subsets(jets):
        color, flabel = flavour_style(fl, i)
        sample_idx = rng.choice(len(fl_jets), size=min(n_jets, len(fl_jets)), replace=False)
        sample_jets = fl_jets.iloc[sample_idx]

        for _, jrow in sample_jets.iterrows():
            eid, jid = jrow["event_id"], jrow["jet_id"]
            jhits = hits[(hits["event_id"] == eid) & (hits["hit_id"] == jid)] \
                    if "hit_id" in hits.columns else \
                    hits[(hits["event_id"] == eid) & (hits["jet_id"] == jid)]

            if jhits.empty:
                continue

            # keep only hits within dR < 0.4 of the jet axis (detector-level cut)
            if "jet_phi" in jrow.index and "jet_eta" in jrow.index:
                jphi = float(jrow["jet_phi"])
                jeta = float(jrow["jet_eta"])
                gx_a = jhits["hit_global_x"].values
                gy_a = jhits["hit_global_y"].values
                gz_a = jhits["hit_global_z"].values
                r_a  = np.hypot(gx_a, gy_a).clip(min=1e-6)
                hit_phi = np.arctan2(gy_a, gx_a)
                hit_eta = np.arcsinh(gz_a / r_a)
                dphi = np.angle(np.exp(1j * (hit_phi - jphi)))
                dR   = np.sqrt(dphi**2 + (hit_eta - jeta)**2)
                jhits = jhits[dR < 0.4]

            if jhits.empty:
                continue

            gx = jhits["hit_global_x"].values
            gy = jhits["hit_global_y"].values
            gz = jhits["hit_global_z"].values
            sd = jhits["hit_sub_det"].values

            # colour hits by sub-detector
            sd_vals = sorted(set(sd))
            sd_cmap = plt.cm.tab10
            sd_colors = {s: sd_cmap(k / max(len(sd_vals) - 1, 1))
                         for k, s in enumerate(sd_vals)}
            hit_colors = [sd_colors[s] for s in sd]

            # --- local coords per layer ---
            jlocal = pd.DataFrame()
            layer_keys = []
            if local:
                jlocal = hits_local[(hits_local["event_id"] == eid) &
                                    (hits_local["jet_id"] == jid)]
                if not jlocal.empty:
                    layer_keys = sorted(
                        jlocal[["hit_sub_det", "hit_layer"]].drop_duplicates().values.tolist()
                    )

            n_local_rows = len(layer_keys)
            n_rows = 1 + n_local_rows          # 1 global row + one row per layer
            n_cols = 3
            fig, axes = plt.subplots(n_rows, n_cols,
                                     figsize=(4 * n_cols, 4 * n_rows),
                                     squeeze=False)

            def scatter(ax, xs, ys, xlabel, ylabel, colors=hit_colors):
                ax.scatter(xs, ys, c=colors, s=10, alpha=0.8)
                ax.set_xlabel(xlabel, fontsize=9)
                ax.set_ylabel(ylabel, fontsize=9)
                ax.set_aspect("equal")

            def draw_jet_axis_global(ax, dx, dy):
                ax.axline((0, 0), (dx, dy), color="red", lw=1,
                          ls="--", alpha=0.7, label="jet axis")

            # row 0: global x-y, x-z, y-z
            scatter(axes[0][0], gx, gy, "x [cm]", "y [cm]")
            scatter(axes[0][1], gx, gz, "x [cm]", "z [cm]")
            scatter(axes[0][2], gy, gz, "y [cm]", "z [cm]")
            axes[0][0].set_title("global x-y", fontsize=8)
            axes[0][1].set_title("global x-z", fontsize=8)
            axes[0][2].set_title("global y-z", fontsize=8)

            if "jet_phi" in jrow.index and "jet_eta" in jrow.index:
                draw_jet_axis_global(axes[0][0], np.cos(jphi), np.sin(jphi))
                draw_jet_axis_global(axes[0][1], np.cos(jphi), np.sinh(jeta))
                draw_jet_axis_global(axes[0][2], np.sin(jphi), np.sinh(jeta))

            # rows 1+: one row per layer with r·Δφ vs r·Δφ⊥, r·Δφ vs Δz, hide 3rd panel
            theta = np.linspace(0, 2 * np.pi, 200)
            for row_idx, (lsd_val, lly) in enumerate(layer_keys):
                layer_hits = jlocal[(jlocal["hit_sub_det"] == lsd_val) &
                                    (jlocal["hit_layer"] == lly)]
                ax_phi = axes[row_idx + 1][0]
                ax_z   = axes[row_idx + 1][1]
                axes[row_idx + 1][2].set_visible(False)

                r_layer = LAYER_RADII.get((lsd_val, lly))
                lcolors = [sd_colors.get(s, "gray") for s in layer_hits["hit_sub_det"]]

                scatter(ax_phi,
                        layer_hits["local_arc"].values,
                        layer_hits["local_dz"].values,
                        "r·Δφ [cm]", "r·Δφ⊥ [cm]", colors=lcolors)
                scatter(ax_z,
                        layer_hits["local_arc"].values,
                        layer_hits["local_z"].values,
                        "r·Δφ [cm]", "Δz [cm]", colors=lcolors)

                sd_name = SUB_DET_NAMES.get(lsd_val, f"sd{lsd_val}")
                ax_phi.set_title(f"{sd_name} L{lly} — r·Δφ vs r·Δφ⊥", fontsize=8)
                ax_z.set_title(f"{sd_name} L{lly} — r·Δφ vs Δz", fontsize=8)

                if r_layer is not None:
                    rc = r_layer * 0.5
                    rc_z = rc * np.cosh(jeta)
                    ax_phi.plot(rc * np.cos(theta), rc * np.sin(theta),
                                color="red", lw=0.8, ls="--", alpha=0.7)
                    ax_z.plot(rc * np.cos(theta), rc_z * np.sin(theta),
                              color="red", lw=0.8, ls="--", alpha=0.7)

            # legend for sub-detectors
            for s, c in sd_colors.items():
                axes[0][0].scatter([], [], c=[c], s=20,
                                   label=SUB_DET_NAMES.get(s, f"sd{s}"))
            axes[0][0].legend(fontsize=7, loc="upper left")

            pt  = jrow.get("jet_pt", float("nan"))
            eta = jrow.get("jet_eta", float("nan"))
            fig.suptitle(f"{flabel}  ev={eid} jet={jid}  "
                         f"pT={pt:.1f} GeV  η={eta:.2f}  n_hits={len(jhits)}",
                         fontsize=10)

            fig.tight_layout()
            fname = f"{fl}_ev{eid}_jet{jid}.png"
            savefig(fig, out_base, fname.replace(".png", ""))


def plot_hit_local_xy(hits, jets, outdir, by_flavour=False, max_range=2.0, bins=80,
                      dr_cone=0.5):
    """
    Plot local (x', y') hit positions in the jet-transverse frame per sub-detector.
    x' = along jet direction, y' = perpendicular (IP direction).
    Uses add_local_layer_coords(use_transverse=True) from jet_layer_intersections.
    """
    from jet_layer_intersections import add_local_layer_coords, LAYER_RADII

    hits_local = add_local_layer_coords(hits, jets,
                                        layer_radii=LAYER_RADII,
                                        use_jet_perp=False)
    # local_arc = x' (perp to jet in transverse plane), local_dz = y' (perp in 3D)
    # also compute local_z = hit_global_z - z_axis for the longitudinal view
    jets_idx = jets[["event_id", "jet_id", "jet_eta", "jet_phi"]]
    hits_local = hits_local.merge(jets_idx, on=["event_id", "jet_id"],
                                  how="left", suffixes=("", "_j"))
    from jet_layer_intersections import LAYER_RADII as _LR
    LAYER_RADII = _LR
    radii_df = pd.DataFrame(
        [(sd, ly, r) for (sd, ly), r in _LR.items()],
        columns=["hit_sub_det", "hit_layer", "layer_radius"],
    )
    hits_local = hits_local.merge(radii_df, on=["hit_sub_det", "hit_layer"],
                                  how="left", suffixes=("", "_r"))
    eta_col = "jet_eta" if "jet_eta" in hits_local.columns else "jet_eta_j"
    hits_local["local_z"] = (hits_local["hit_global_z"]
                             - hits_local["layer_radius"] * np.sinh(hits_local[eta_col]))

    sub_dets = sorted(hits_local["hit_sub_det"].unique())

    for sd in sub_dets:
        sd_hits = hits_local[hits_local["hit_sub_det"] == sd]
        layers  = sorted(sd_hits["hit_layer"].unique())
        n_layers = len(layers)
        if n_layers == 0:
            continue

        if by_flavour:
            flavours_present = sorted(sd_hits["jet_label"].dropna().unique()) \
                               if "jet_label" in sd_hits.columns else []
            n_fl = len(flavours_present)
            if n_fl == 0:
                by_flavour_here = False
            else:
                by_flavour_here = True
        else:
            by_flavour_here = False

        # two panels (r·Δφ vs Δz) per flavour: azimuthal and longitudinal cone projections
        n_cols = n_fl * 2 if by_flavour_here else 2
        fig, axes = plt.subplots(n_layers, n_cols,
                                 figsize=(4 * n_cols, 4 * n_layers),
                                 squeeze=False)

        # mean cosh(eta) over jets in this sub-det to set z extent of cone ellipse
        eta_col = "jet_eta" if "jet_eta" in sd_hits.columns else "jet_eta_j"
        mean_cosh_eta = float(np.cosh(sd_hits[eta_col].abs()).mean()) if eta_col in sd_hits.columns else 1.0
        theta = np.linspace(0, 2 * np.pi, 200)

        def _draw_cone_ellipse(ax_phi, ax_z, rc, cosh_eta):
            """Elliptical cone boundary: semi-axis rc in Δφ·r, rc·cosh(η) in Δz."""
            rc_z = rc * cosh_eta
            ax_phi.plot(rc * np.cos(theta), rc * np.sin(theta),
                        color="red", lw=0.8, ls="--", alpha=0.7)
            ax_z.plot(rc * np.cos(theta), rc_z * np.sin(theta),
                      color="red", lw=0.8, ls="--", alpha=0.7)

        for row, layer in enumerate(layers):
            layer_hits = sd_hits[sd_hits["hit_layer"] == layer]
            r_layer = LAYER_RADII.get((sd, layer), None)
            rc = r_layer * dr_cone if r_layer is not None else max_range
            rc_z = rc * mean_cosh_eta
            phi_range = max(max_range, rc * 1.2)
            z_range   = max(max_range, rc_z * 1.2)

            if by_flavour_here:
                for col, (i, fl, sub) in enumerate(
                        flavour_subsets(layer_hits, label_col="jet_label")):
                    ax_phi = axes[row][col * 2]
                    ax_z   = axes[row][col * 2 + 1]
                    color, label = flavour_style(fl, i)
                    ax_phi.hist2d(sub["local_arc"], sub["local_dz"],
                                  bins=bins,
                                  range=[[-phi_range, phi_range], [-phi_range, phi_range]],
                                  cmap="Blues")
                    ax_phi.set_title(f"sd{sd} L{layer} {label} — r·Δφ vs r·Δφ⊥", fontsize=9)
                    ax_phi.set_xlabel("r·Δφ [cm]", fontsize=8)
                    ax_phi.set_ylabel("r·Δφ⊥ [cm]", fontsize=8)
                    ax_phi.set_aspect("equal")
                    ax_z.hist2d(sub["local_arc"], sub["local_z"],
                                bins=bins,
                                range=[[-phi_range, phi_range], [-z_range, z_range]],
                                cmap="Oranges")
                    ax_z.set_title(f"sd{sd} L{layer} {label} — r·Δφ vs Δz", fontsize=9)
                    ax_z.set_xlabel("r·Δφ [cm]", fontsize=8)
                    ax_z.set_ylabel("Δz [cm]", fontsize=8)
                    ax_z.set_aspect("equal")
                    if r_layer is not None:
                        _draw_cone_ellipse(ax_phi, ax_z, rc, mean_cosh_eta)
            else:
                ax_phi = axes[row][0]
                ax_z   = axes[row][1]
                ax_phi.hist2d(layer_hits["local_arc"], layer_hits["local_dz"],
                              bins=bins,
                              range=[[-phi_range, phi_range], [-phi_range, phi_range]],
                              cmap="Blues")
                ax_phi.set_title(f"sd{sd} layer {layer} — r·Δφ vs r·Δφ⊥", fontsize=9)
                ax_phi.set_xlabel("r·Δφ [cm]", fontsize=8)
                ax_phi.set_ylabel("r·Δφ⊥ [cm]", fontsize=8)
                ax_phi.set_aspect("equal")
                ax_z.hist2d(layer_hits["local_arc"], layer_hits["local_z"],
                            bins=bins,
                            range=[[-phi_range, phi_range], [-z_range, z_range]],
                            cmap="Oranges")
                ax_z.set_title(f"sd{sd} layer {layer} — r·Δφ vs Δz", fontsize=9)
                ax_z.set_xlabel("r·Δφ [cm]", fontsize=8)
                ax_z.set_ylabel("Δz [cm]", fontsize=8)
                ax_z.set_aspect("equal")
                if r_layer is not None:
                    _draw_cone_ellipse(ax_phi, ax_z, rc, mean_cosh_eta)

        sd_name = SUB_DET_NAMES.get(sd, f"sd{sd}")
        fig.suptitle(f"Local hit positions — {sd_name}"
                     + (" by flavour" if by_flavour_here else ""), fontsize=11)
        fig.tight_layout()
        suffix = f"_by_flavour" if by_flavour_here else ""
        savefig(fig, outdir, f"hit_local_xy_sd{sd}{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", help="Path to hits_jets.h5 produced by extract_hit_jet_association.py")
    parser.add_argument("--outdir", default="plots", help="Directory to write plots into")
    parser.add_argument("--by-flavour", action="store_true",
                        help="Overlay b/c/light flavours on each plot")
    parser.add_argument("--n-single-jets", type=int, default=5,
                        help="Number of random jets per flavour to plot individually (0 = skip)")
    parser.add_argument("--single-jet-seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    hits = pd.read_hdf(args.input_h5, key="hits")
    jets = pd.read_hdf(args.input_h5, key="jets")

    print(f"Loaded {len(hits)} hits and {len(jets)} jets")

    if args.by_flavour:
        hits = merge_flavour(hits, jets)

    print("Plotting jet kinematics...")
    plot_jet_kinematics(jets, args.outdir, by_flavour=args.by_flavour)
    print("Plotting n_hits per jet...")
    plot_n_hits_per_jet(jets, args.outdir, by_flavour=args.by_flavour)
    n_jets = len(jets)
    print("Plotting hits per layer...")
    plot_hits_per_layer(hits, args.outdir, n_jets)
    print("Plotting hits per global layer...")
    plot_hits_per_global_layer(hits, args.outdir, n_jets, by_flavour=args.by_flavour)
    print("Plotting average within-jet hit distance per layer...")
    plot_avg_hit_distance_per_layer(hits, args.outdir, by_flavour=args.by_flavour)
    print("Plotting hit positions...")
    plot_hit_positions(hits, args.outdir, by_flavour=args.by_flavour)
    print("Plotting local x/y hit positions...")
    plot_hit_local_xy(hits, jets, args.outdir, by_flavour=args.by_flavour)

    if args.n_single_jets > 0:
        print(f"Plotting {args.n_single_jets} random jets per flavour...")
        plot_single_jets(hits, jets, args.outdir, n_jets=args.n_single_jets,
                         seed=args.single_jet_seed)

    print("Done.")


if __name__ == "__main__":
    main()
