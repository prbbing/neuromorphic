"""
Build a sparse spike-list event representation of jets, suitable as input to
event-based / spiking neural network pipelines (the same (t, x, y) format
used for DVS camera data).

For every hit on a jet:
  t = compact layer index -- the global_layer (sub-detector-spanning layer
                        index, see analyze_hits_jets.global_layer_offsets)
                        remapped to consecutive integers 0..12, dropping the
                        endcap-only slots (PixelEndcap, TID, TEC) that never
                        appear in this barrel-only dataset. Used as the
                        discrete timestep (particles cross layers in order as
                        they propagate outward).
  x = local_arc      -- R * delta_phi relative to the jet's own predicted
                         axis-crossing point on that layer (see
                         jet_layer_intersections.add_local_layer_coords).
  y = local_dz        -- delta_z relative to the same axis-crossing point.

Each row also carries event_id, jet_id, and jet_label so events can be
grouped back into per-jet sequences and used for classification.

Usage:
  python3 build_event_list.py hits_jets_qcd.h5 hits_jets_ttbar.h5 \
      --labels QCD ttbar -o events.h5
"""

import argparse

import numpy as np
import pandas as pd

from jet_layer_intersections import add_local_layer_coords, LAYER_RADII
from analyze_hits_jets import add_global_layer, global_layer_offsets, SUB_DET_N_LAYERS

# sub-detectors covered by LAYER_RADII (i.e. the ones with hits in this barrel-only dataset)
KNOWN_SUB_DETS = sorted({sd for sd, _ in LAYER_RADII})


def compact_layer_mapping(sub_dets=KNOWN_SUB_DETS):
    """Map global_layer -> consecutive 0..N-1 index, covering only the given
    sub-detectors' layers (in physical order), so gaps from excluded
    sub-detectors (e.g. endcaps) collapse out of the timestep axis."""
    offsets = global_layer_offsets()
    global_layers = sorted(
        layer + offsets[sd] for sd in sub_dets for layer in range(1, SUB_DET_N_LAYERS[sd] + 1)
    )
    return {g: i for i, g in enumerate(global_layers)}


def build_events(hits, jets, jet_label=None, layer_mapping=None,
                 use_jet_perp=True, use_transverse=False):
    hits_local = add_local_layer_coords(hits, jets, layer_radii=LAYER_RADII,
                                        use_jet_perp=use_jet_perp,
                                        use_transverse=use_transverse)
    hits_local["global_layer"] = add_global_layer(hits_local)
    if layer_mapping is None:
        layer_mapping = compact_layer_mapping()
    hits_local["t"] = hits_local["global_layer"].map(layer_mapping)

    events = hits_local[["event_id", "jet_id", "t", "local_arc", "local_dz"]].rename(
        columns={"local_arc": "x", "local_dz": "y"}
    )

    # attach per-jet properties: label, eta, pt
    jet_cols = ["event_id", "jet_id"]
    for col in ["jet_label", "jet_eta", "jet_pt"]:
        if col in jets.columns:
            jet_cols.append(col)
    events = events.merge(jets[jet_cols], on=["event_id", "jet_id"], how="left")
    if jet_label is not None:
        events["jet_label"] = jet_label

    return events.sort_values(["event_id", "jet_id", "t"]).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_h5", nargs="+", help="One or more hits_jets*.h5 files")
    parser.add_argument("--labels", nargs="+", default=None,
                         help="Override jet_label for each input file (default: use the "
                              "jet_label already stored in each file's jets table)")
    parser.add_argument("-o", "--output", default="events.h5", help="Output .h5 path")
    parser.add_argument("--no-jet-perp", action="store_true",
                         help="Use original cylindrical (local_arc, local_dz) coordinates "
                              "instead of the jet-perpendicular plane projection (default: off)")
    parser.add_argument("--transverse", action="store_true",
                         help="Use transverse-plane jet-rotated coordinates: rotate global "
                              "(x,y) by -phi_jet so the jet points along +x, then subtract "
                              "the jet crossing (R,0). Z is dropped. Overrides --no-jet-perp.")
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) != len(args.input_h5):
        parser.error("--labels must have the same number of entries as input_h5 files")

    layer_mapping = compact_layer_mapping()
    print(f"compact timestep mapping (global_layer -> t): {layer_mapping}")

    all_events = []
    for i, path in enumerate(args.input_h5):
        hits = pd.read_hdf(path, key="hits")
        jets = pd.read_hdf(path, key="jets")
        label = args.labels[i] if args.labels is not None else None
        events = build_events(hits, jets, jet_label=label, layer_mapping=layer_mapping,
                              use_jet_perp=not args.no_jet_perp,
                              use_transverse=args.transverse)
        # disambiguate event_id/jet_id across input files by prefixing with a source index
        events["jet_uid"] = f"{i}_" + events["event_id"].astype(str) + "_" + events["jet_id"].astype(str)
        n_jets = events[["event_id", "jet_id"]].drop_duplicates().shape[0]
        print(f"{path}: {len(events)} hits from {n_jets} jets "
              f"(label={events['jet_label'].iloc[0] if 'jet_label' in events else 'n/a'})")
        all_events.append(events)

    out = pd.concat(all_events, ignore_index=True)
    out.to_hdf(args.output, key="events", mode="w", format="table")
    print(f"Wrote {len(out)} total events to {args.output}")


if __name__ == "__main__":
    main()
