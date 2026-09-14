"""
Extract hit-to-jet associations from CMS Open Data tracker-hit ntuples
(e.g. QCD_Pt-15to3000_TuneZ2star_Flat_8TeV_pythia6 or ttbar samples,
tracker-hit-enriched). Use --jet-label to set the truth label written to
the output for a given input file/sample.

For every event, hits are associated to gen jets geometrically: each hit is
assigned to the nearest gen jet within dR < dr_cone (default 0.5, matching the
AK5 jet radius), computed from the hit's global (x, y, z) position. Only jets
with pT >= min_jet_pt (default 50 GeV) and |eta| <= max_jet_eta (default 2.0)
are considered. Hits outside any jet cone, and hits whose nearest jet fails the
pT/eta selection, are dropped.

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

Truth-flavor labeling (--truth-flavor, for ttbar samples):
  Reads gen-particle branches to identify the hard-scatter top-quark decay chain.
  For each event:
    1. Locate top quarks (|pdgid|=6, status=3) -- two per ttbar event.
    2. Find b quarks (|pdgid|=5, status=3) and match each to the nearest top via
       deltaR; these are the b quarks from t -> W b.
    3. Find W bosons (|pdgid|=24, status=3) and match each to the nearest top.
    4. Find light quarks (|pdgid| in 1-4, status=3) that are within deltaR < 1.0
       of a W boson; these are the q q' daughters of W -> q q'.
    5. For each gen jet, find the closest hard-scatter parton (b or W-daughter
       light quark). If deltaR < dr_match (default 0.4), the jet is labeled:
         jet_label        = "b"     (if matched to a b quark, |pdgid|=5)
         jet_label        = "c"     (if matched to a c quark, |pdgid|=4, from W decay)
         jet_label        = "light" (if matched to a light quark, |pdgid|=1-3, from W decay)
         jet_parton_pdgid = pdgid of the matched parton
         top_index        = 0 or 1 (which of the two tops in the event)
    6. Jets not matched to any top-decay parton are dropped.
  Note: the ntuple stores genpart_pz under a duplicate "genpart_px" branch name
  at branch position 35 (a known mislabelling). This script reads it by position.
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


def _delta_r(eta1, phi1, eta2, phi2):
    dphi = (phi1 - phi2 + np.pi) % (2 * np.pi) - np.pi
    return np.sqrt((eta1 - eta2) ** 2 + dphi ** 2)


def _match_jets_to_partons(jet_eta, jet_phi, parton_eta, parton_phi,
                           parton_pdgid, parton_top_idx, dr_max=0.4):
    """Return per-jet arrays: matched_flavor, matched_pdgid, matched_top_idx.

    matched_flavor is 'b', 'c', 'light', or None.
    Unmatched jets have None / pdgid=0 / top_idx=-1.
    """
    n_jets = len(jet_eta)
    flavors   = [None] * n_jets
    pdgids    = np.zeros(n_jets, dtype=np.int32)
    top_idxs  = np.full(n_jets, -1, dtype=np.int32)

    if len(parton_eta) == 0:
        return flavors, pdgids, top_idxs

    for ji in range(n_jets):
        drs = _delta_r(jet_eta[ji], jet_phi[ji], parton_eta, parton_phi)
        best = np.argmin(drs)
        if drs[best] < dr_max:
            pid = parton_pdgid[best]
            apid = abs(pid)
            flavors[ji]  = "b" if apid == 5 else ("c" if apid == 4 else "light")
            pdgids[ji]   = pid
            top_idxs[ji] = parton_top_idx[best]

    return flavors, pdgids, top_idxs


def _remove_overlapping_jets(jet_indices, jet_eta, jet_phi, jet_pt, dr_min=0.5):
    """Greedy overlap removal: keep jets sorted by pT desc, drop any that overlap
    (deltaR < dr_min) with an already-kept jet. Returns array of kept indices."""
    order = np.argsort(jet_pt[jet_indices])[::-1]
    kept = []
    kept_eta, kept_phi = [], []
    for i in order:
        ji = jet_indices[i]
        if kept_eta:
            drs = _delta_r(jet_eta[ji], jet_phi[ji],
                           np.array(kept_eta), np.array(kept_phi))
            if drs.min() < dr_min:
                continue
        kept.append(ji)
        kept_eta.append(jet_eta[ji])
        kept_phi.append(jet_phi[ji])
    return np.array(kept, dtype=np.intp)


def _build_top_decay_partons(pdgid, status, px, py, pz, w_dr_max=1.0):
    """Extract hard-scatter top-decay partons from a single event's genpart arrays.

    Returns (parton_eta, parton_phi, parton_pdgid, parton_top_idx) as numpy arrays.
    parton_top_idx is 0 or 1, ordered by top pT descending.
    """
    st3 = status == 3
    pt_all = np.hypot(px, py)
    eta_all = np.arcsinh(pz / np.where(pt_all > 1e-6, pt_all, 1e-6))
    phi_all = np.arctan2(py, px)

    top_mask  = st3 & (np.abs(pdgid) == 6)
    b_mask    = st3 & (np.abs(pdgid) == 5)
    W_mask    = st3 & (np.abs(pdgid) == 24)
    q_mask    = st3 & (np.abs(pdgid) >= 1) & (np.abs(pdgid) <= 4)

    top_idx_arr = np.where(top_mask)[0]
    if len(top_idx_arr) < 1:
        return np.array([]), np.array([]), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    # order tops by pT descending so top_index=0 is always the harder top
    top_idx_arr = top_idx_arr[np.argsort(pt_all[top_idx_arr])[::-1]]
    top_eta = eta_all[top_idx_arr]
    top_phi = phi_all[top_idx_arr]
    n_tops  = len(top_idx_arr)

    p_eta_list, p_phi_list, p_pid_list, p_top_list = [], [], [], []

    # b quarks: match to nearest top
    for bi in np.where(b_mask)[0]:
        drs = _delta_r(eta_all[bi], phi_all[bi], top_eta, top_phi)
        ti  = int(np.argmin(drs))
        p_eta_list.append(eta_all[bi]); p_phi_list.append(phi_all[bi])
        p_pid_list.append(pdgid[bi]);   p_top_list.append(ti)

    # W bosons: match to nearest top
    W_indices  = np.where(W_mask)[0]
    W_top_map  = {}  # W array-index -> top_index
    for wi in W_indices:
        drs = _delta_r(eta_all[wi], phi_all[wi], top_eta, top_phi)
        W_top_map[wi] = int(np.argmin(drs))

    # light quarks from W: keep only those within w_dr_max of a W
    for qi in np.where(q_mask)[0]:
        if len(W_indices) == 0:
            continue
        drs_W = _delta_r(eta_all[qi], phi_all[qi], eta_all[W_indices], phi_all[W_indices])
        best_w = int(np.argmin(drs_W))
        if drs_W[best_w] < w_dr_max:
            wi = W_indices[best_w]
            ti = W_top_map[wi]
            p_eta_list.append(eta_all[qi]); p_phi_list.append(phi_all[qi])
            p_pid_list.append(pdgid[qi]);   p_top_list.append(ti)

    if not p_eta_list:
        return np.array([]), np.array([]), np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    return (np.array(p_eta_list), np.array(p_phi_list),
            np.array(p_pid_list, dtype=np.int32), np.array(p_top_list, dtype=np.int32))


def latest_cycle_key(file_obj, base_name):
    cycles = [k for k in file_obj.keys() if k.split(";")[0] == base_name]
    if not cycles:
        raise KeyError(f"No tree named '{base_name}' found in file")
    return max(cycles, key=lambda k: int(k.split(";")[1]))


GENPART_BRANCHES = ["genpart_pdgid", "genpart_status", "genpart_px", "genpart_py", "genpart_energy"]


def extract(root_path, tree_name="hits_tree", max_events=None, min_jet_pt=50.0, max_jet_eta=2.0,
            jet_label="QCD", barrel_only=False, hit_types=DEFAULT_HIT_TYPES,
            truth_flavor=False, dr_match=0.4, dr_cone=0.5):
    f = uproot.open(root_path)
    key = latest_cycle_key(f, tree_name)
    tree = f[key]

    branches = ["hit_genjet_id"] + HIT_FIELDS + JET_FIELDS
    arrs = tree.arrays(branches, entry_stop=max_events, library="np")

    if truth_flavor:
        gp_arrs = tree.arrays(GENPART_BRANCHES, entry_stop=max_events, library="np")
        # genpart_pz is mislabelled as a second "genpart_px" at branch position 35
        gp_pz = list(tree.values())[35].array(entry_stop=max_events, library="np")

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

        # truth-flavor labeling: match gen jets to top-decay partons
        if truth_flavor:
            gp_pdgid  = np.asarray(gp_arrs["genpart_pdgid"][ev])
            gp_status = np.asarray(gp_arrs["genpart_status"][ev])
            gp_px     = np.asarray(gp_arrs["genpart_px"][ev])
            gp_py     = np.asarray(gp_arrs["genpart_py"][ev])
            gp_pz_ev  = np.asarray(gp_pz[ev])

            p_eta, p_phi, p_pid, p_top = _build_top_decay_partons(
                gp_pdgid, gp_status, gp_px, gp_py, gp_pz_ev)
            jet_flavors, jet_pdgids, jet_top_idxs = _match_jets_to_partons(
                jet_eta, jet_phi, p_eta, p_phi, p_pid, p_top, dr_max=dr_match)

            # only keep jets matched to a top-decay parton
            jet_keep = np.array([f is not None for f in jet_flavors])
        else:
            jet_keep = np.ones(n_jets, dtype=bool)

        jet_ok = (jet_pt >= min_jet_pt) & (np.abs(jet_eta) <= max_jet_eta) & jet_keep

        # remove overlapping jets (dR < 0.5), keeping higher-pT jet
        ok_indices = np.where(jet_ok)[0]
        if len(ok_indices) > 1:
            kept = _remove_overlapping_jets(ok_indices, jet_eta, jet_phi, jet_pt)
            jet_ok[:] = False
            jet_ok[kept] = True

        # geometric hit-to-jet association: assign each hit to the nearest jet
        # within dr_cone (AK5 radius = 0.5), ignoring truth links
        hx  = np.asarray(arrs["hit_global_x"][ev])
        hy  = np.asarray(arrs["hit_global_y"][ev])
        hz  = np.asarray(arrs["hit_global_z"][ev])
        hr  = np.hypot(hx, hy).clip(min=1e-6)
        h_phi = np.arctan2(hy, hx)
        h_eta = np.arcsinh(hz / hr)

        n_hits = len(hx)
        hit_jet_id = np.full(n_hits, -1, dtype=int)
        hit_min_dr = np.full(n_hits, np.inf)

        for jidx in np.where(jet_ok)[0]:
            dphi = np.angle(np.exp(1j * (h_phi - jet_phi[jidx])))
            dR   = np.sqrt(dphi**2 + (h_eta - jet_eta[jidx])**2)
            closer = dR < np.minimum(hit_min_dr, dr_cone)
            hit_jet_id[closer] = jidx
            hit_min_dr[closer] = dR[closer]

        match = hit_jet_id >= 0

        if barrel_only:
            hit_sub_det = np.asarray(arrs["hit_sub_det"][ev])
            match &= np.isin(hit_sub_det, BARREL_SUB_DETS)

        if hit_types is not None:
            hit_type = np.asarray(arrs["hit_type"][ev])
            match &= np.isin(hit_type, hit_types)

        if match.any():
            gid = hit_jet_id[match]
            hit_df = {"event_id": ev, "jet_id": gid}
            for field in HIT_FIELDS:
                hit_df[field] = np.asarray(arrs[field][ev])[match]
            hit_rows.append(pd.DataFrame(hit_df))
        else:
            gid = np.array([], dtype=gid_all.dtype)

        n_hits_per_jet = np.bincount(gid, minlength=n_jets) if n_jets else np.zeros(0, dtype=int)
        has_hits = (n_hits_per_jet > 0) & jet_ok

        sel = np.where(has_hits)[0]
        jet_df = {
            "event_id": ev,
            "jet_id": sel,
            "jet_pt": jet_pt[sel],
            "jet_eta": jet_eta[sel],
            "jet_phi": jet_phi[sel],
            "n_hits": n_hits_per_jet[sel],
            "jet_label": [jet_flavors[j] if truth_flavor else jet_label for j in sel],
        }
        if truth_flavor:
            jet_df["jet_parton_pdgid"] = [int(jet_pdgids[j])   for j in sel]
            jet_df["top_index"]        = [int(jet_top_idxs[j]) for j in sel]
        jet_rows.append(pd.DataFrame(jet_df))

    hits_df = pd.concat(hit_rows, ignore_index=True) if hit_rows else pd.DataFrame()
    jets_df = pd.concat(jet_rows, ignore_index=True) if jet_rows else pd.DataFrame()
    return hits_df, jets_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_root", nargs="+", help="One or more input .root files")
    parser.add_argument("output_h5", help="Path to output .h5 file")
    parser.add_argument("--max-events", type=int, default=None,
                         help="Limit number of events processed per file (for testing)")
    parser.add_argument("--min-jet-pt", type=float, default=50.0,
                         help="Minimum jet pT [GeV] to keep (default: 50)")
    parser.add_argument("--max-jet-eta", type=float, default=2.0,
                         help="Maximum |jet eta| to keep, restricts jets to the "
                              "tracker barrel acceptance (default: 2.0)")
    parser.add_argument("--jet-label", default="QCD",
                         help="Label to assign to all jets, e.g. 'QCD' or 'ttbar' "
                              "(default: QCD)")
    parser.add_argument("--barrel-only", action="store_true",
                         help="Additionally drop hits outside the barrel sub-detectors "
                              "(keep only hit_sub_det in {1 PixelBarrel, 3 TIB, 5 TOB})")
    parser.add_argument("--hit-types", type=int, nargs="*", default=list(DEFAULT_HIT_TYPES),
                         help="hit_type values to keep; removes exact-duplicate hit_type 3/4 "
                              "records and collapses stereo-module pairs (1,2) down to one "
                              "representative hit per layer (default: 0 1). Pass --hit-types "
                              "with no values to disable this filter.")
    parser.add_argument("--truth-flavor", action="store_true",
                         help="For ttbar samples: label each jet as 'b', 'c', or 'light' based "
                              "on deltaR matching to hard-scatter top-decay partons (b from top, "
                              "c/light from W). Adds jet_parton_pdgid and top_index columns. "
                              "Jets not matched to any top-decay parton are dropped.")
    parser.add_argument("--dr-match", type=float, default=0.4,
                         help="Max deltaR for parton-to-jet matching when --truth-flavor is "
                              "used (default: 0.4)")
    parser.add_argument("--dr-cone", type=float, default=0.5,
                         help="Cone size for geometric hit-to-jet association; matches the "
                              "AK5 gen jet radius (default: 0.5)")
    args = parser.parse_args()

    hit_types = tuple(args.hit_types) if args.hit_types else None

    all_hits, all_jets = [], []
    for i, root_path in enumerate(args.input_root):
        print(f"[{i+1}/{len(args.input_root)}] Processing {root_path}...")
        hits_df, jets_df = extract(root_path, max_events=args.max_events,
                                    min_jet_pt=args.min_jet_pt, max_jet_eta=args.max_jet_eta,
                                    jet_label=args.jet_label, barrel_only=args.barrel_only,
                                    hit_types=hit_types,
                                    truth_flavor=args.truth_flavor, dr_match=args.dr_match,
                                    dr_cone=args.dr_cone)
        # prefix event_id with file index to avoid collisions across files
        hits_df["event_id"] = hits_df["event_id"].astype(str) + f"_f{i}"
        jets_df["event_id"] = jets_df["event_id"].astype(str) + f"_f{i}"
        print(f"  {len(hits_df)} hits, {len(jets_df)} jets")
        all_hits.append(hits_df)
        all_jets.append(jets_df)

    hits_out = pd.concat(all_hits, ignore_index=True)
    jets_out = pd.concat(all_jets, ignore_index=True)
    print(f"Total: {len(hits_out)} hits, {len(jets_out)} jets across {len(args.input_root)} files")

    hits_out.to_hdf(args.output_h5, key="hits", mode="w", format="table")
    jets_out.to_hdf(args.output_h5, key="jets", mode="a", format="table")
    print(f"Wrote 'hits' and 'jets' tables to {args.output_h5}")


if __name__ == "__main__":
    main()
