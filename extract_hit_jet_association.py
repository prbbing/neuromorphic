"""
Extract hit-to-jet associations from CMS Open Data tracker-hit ntuples
(e.g. QCD_Pt-15to3000_TuneZ2star_Flat_8TeV_pythia6 or ttbar samples,
tracker-hit-enriched). Use --jet-label to set the truth label written to
the output for a given input file/sample.

For every event, every hit that is matched to a gen jet (hit_genjet_match == True)
with jet pT >= min_jet_pt (default 200 GeV) and |jet eta| <= max_jet_eta (default
2.0, to keep jets within the tracker barrel acceptance) is written out with its
own features plus (event_id, jet_id) keys. Unmatched hits, and hits matched to
jets failing the pT or eta selection, are dropped.

Note that a jet-level eta cut alone does not guarantee all of its hits land in
the barrel: individual hits within a jet's cone can fan out to a different eta
than the jet axis and land in endcap sub-detectors (hit_sub_det 2 PixelEndcap,
4 TID, 6 TEC). Pass --barrel-only to additionally drop those hits directly,
keeping only hit_sub_det in {1 PixelBarrel, 3 TIB, 5 TOB}.

Strip tracker layers record every physical hit twice at the same global
position, tagged with paired hit_type values: (1, 3) or (2, 4). hit_type 3/4
are exact-duplicate copies and are always dropped.

Additionally, TIB/TOB layers 1-2 use double-sided (stereo) modules: each
particle crossing produces two genuine but very closely-spaced measurements
(hit_type 1 and hit_type 2, an r-phi hit and a stereo hit a few mm apart),
unlike the single-sided layers further out which only ever produce hit_type 1.
To keep one representative hit per physical particle crossing throughout
(rather than two near-duplicate stereo hits in layers 1-2), only hit_type in
{0, 1} is kept by default (--hit-types to override).

A second table holds one row per gen jet that has at least one matched hit
(jets with zero matched hits are dropped), keyed by the same (event_id, jet_id),
so the two tables can be joined/concatenated as needed:
hits.merge(jets, on=["event_id", "jet_id"]).

Output: two HDF5 tables ("hits" and "jets") written with pandas, in one .h5 file.

Note: jet_id (== hit_genjet_id) is simply the 0-based index into that event's
genjet_* arrays -- it is local to the event, not a global jet ID.
"""

import argparse
import numpy as np
import pandas as pd
import uproot


HIT_FIELDS = [
    "hit_global_x", "hit_global_y", "hit_global_z",
    "hit_local_x", "hit_local_y",
    "hit_local_x_error", "hit_local_y_error",
    "hit_sub_det", "hit_layer", "hit_type",
]

JET_FIELDS = [
    "genjet_px", "genjet_py", "genjet_pz", "genjet_energy",
]

BARREL_SUB_DETS = (1, 3, 5)  # PixelBarrel, TIB, TOB
DEFAULT_HIT_TYPES = (0, 1)  # excludes hit_type 3/4 (exact duplicates) and 2 (stereo-pair partner of 1)


def jet_pt_eta_phi(px, py, pz):
    pt = np.hypot(px, py)
    p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
    eta = np.arctanh(np.clip(pz / np.where(p == 0, 1, p), -1 + 1e-9, 1 - 1e-9))
    phi = np.arctan2(py, px)
    return pt, eta, phi


def latest_cycle_key(file_obj, base_name):
    cycles = [k for k in file_obj.keys() if k.split(";")[0] == base_name]
    if not cycles:
        raise KeyError(f"No tree named '{base_name}' found in file")
    return max(cycles, key=lambda k: int(k.split(";")[1]))


def extract(root_path, tree_name="hits_tree", max_events=None, min_jet_pt=200.0, max_jet_eta=2.0,
            jet_label="QCD", barrel_only=False, hit_types=DEFAULT_HIT_TYPES):
    f = uproot.open(root_path)
    key = latest_cycle_key(f, tree_name)
    tree = f[key]

    branches = ["hit_genjet_id", "hit_genjet_match"] + HIT_FIELDS + JET_FIELDS
    arrs = tree.arrays(branches, entry_stop=max_events, library="np")

    n_events = len(arrs["hit_genjet_id"])
    hit_rows = []
    jet_rows = []

    for ev in range(n_events):
        gid_all = np.asarray(arrs["hit_genjet_id"][ev])

        jet_px = np.asarray(arrs["genjet_px"][ev])
        jet_py = np.asarray(arrs["genjet_py"][ev])
        jet_pz = np.asarray(arrs["genjet_pz"][ev])
        jet_pt, jet_eta, jet_phi = jet_pt_eta_phi(jet_px, jet_py, jet_pz)
        n_jets = len(jet_px)

        jet_ok = (jet_pt >= min_jet_pt) & (np.abs(jet_eta) <= max_jet_eta)
        match = np.asarray(arrs["hit_genjet_match"][ev], dtype=bool) & jet_ok[gid_all]

        if barrel_only:
            hit_sub_det = np.asarray(arrs["hit_sub_det"][ev])
            match &= np.isin(hit_sub_det, BARREL_SUB_DETS)

        if hit_types is not None:
            hit_type = np.asarray(arrs["hit_type"][ev])
            match &= np.isin(hit_type, hit_types)

        if match.any():
            gid = gid_all[match]
            hit_df = {"event_id": ev, "jet_id": gid}
            for field in HIT_FIELDS:
                hit_df[field] = np.asarray(arrs[field][ev])[match]
            hit_rows.append(pd.DataFrame(hit_df))
        else:
            gid = np.array([], dtype=gid_all.dtype)

        n_hits_per_jet = np.bincount(gid, minlength=n_jets) if n_jets else np.zeros(0, dtype=int)
        has_hits = (n_hits_per_jet > 0) & jet_ok

        jet_df = {
            "event_id": ev,
            "jet_id": np.arange(n_jets)[has_hits],
            "jet_pt": jet_pt[has_hits],
            "jet_eta": jet_eta[has_hits],
            "jet_phi": jet_phi[has_hits],
            "n_hits": n_hits_per_jet[has_hits],
            "jet_label": jet_label,
        }
        jet_rows.append(pd.DataFrame(jet_df))

    hits_df = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    jets_df = pd.concat(jet_rows, ignore_index=True) if jet_rows else pd.DataFrame()
    return hits_df, jets_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", help="Path to input .root file")
    parser.add_argument("output_h5", help="Path to output .h5 file")
    parser.add_argument("--max-events", type=int, default=None,
                         help="Limit number of events processed (for testing)")
    parser.add_argument("--min-jet-pt", type=float, default=200.0,
                         help="Minimum jet pT [GeV] to keep (default: 200)")
    parser.add_argument("--max-jet-eta", type=float, default=2.0,
                         help="Maximum |jet eta| to keep, restricts jets to the "
                              "tracker barrel acceptance (default: 2.0)")
    parser.add_argument("--jet-label", default="QCD",
                         help="Label to assign to all jets in this file, e.g. 'QCD' or 'ttbar' "
                              "(default: QCD)")
    parser.add_argument("--barrel-only", action="store_true",
                         help="Additionally drop hits outside the barrel sub-detectors "
                              "(keep only hit_sub_det in {1 PixelBarrel, 3 TIB, 5 TOB})")
    parser.add_argument("--hit-types", type=int, nargs="*", default=list(DEFAULT_HIT_TYPES),
                         help="hit_type values to keep; removes exact-duplicate hit_type 3/4 "
                              "records and collapses stereo-module pairs (1,2) down to one "
                              "representative hit per layer (default: 0 1). Pass --hit-types "
                              "with no values to disable this filter.")
    args = parser.parse_args()

    hit_types = tuple(args.hit_types) if args.hit_types else None

    hits_df, jets_df = extract(args.input_root, max_events=args.max_events,
                                min_jet_pt=args.min_jet_pt, max_jet_eta=args.max_jet_eta,
                                jet_label=args.jet_label, barrel_only=args.barrel_only,
                                hit_types=hit_types)
    print(f"Extracted {len(hits_df)} matched hits and {len(jets_df)} gen jets "
          f"across {jets_df['event_id'].nunique() if len(jets_df) else 0} events")

    hits_df.to_hdf(args.output_h5, key="hits", mode="w", format="table")
    jets_df.to_hdf(args.output_h5, key="jets", mode="a", format="table")
    print(f"Wrote 'hits' and 'jets' tables to {args.output_h5}")


if __name__ == "__main__":
    main()
