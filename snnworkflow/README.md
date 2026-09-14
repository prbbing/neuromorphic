# Neuromorphic Jet Classification Pipeline

Spiking neural network (SNN) pipeline for QCD vs ttbar jet classification using CMS tracker hit data from [CERN Open Data record 12220](https://opendata.cern.ch/record/12220).

The pipeline converts raw tracker hits into a DVS-style sparse event representation `(t, x, y)` — where `t` is the layer index (timestep), `x` is the arc-length offset from the jet axis, and `y` is the z-offset — then bins them into per-jet image frames and trains an snnTorch spiking CNN.

---

## Pipeline Overview

```
ROOT file
    │
    ▼
extract_hit_jet_association.py   →   hits_jets_<label>.h5
    │
    ├──► analyze_hits_jets.py        (single-file diagnostics)
    ├──► compare_hits_jets.py        (QCD vs ttbar comparison plots)
    └──► jet_layer_intersections.py  (axis intersection & jet shape plots)
    │
    ▼
build_event_list.py              →   events.h5
    │
    ▼
bin_events_to_frames.py          →   frames.npz
    │
    ▼
train_snn.py                     →   trained model + val accuracy
```

---

## Dependencies

```bash
pip install uproot numpy pandas matplotlib scipy h5py tables snntorch torch
```

---

## Scripts

### 1. `extract_hit_jet_association.py`

Reads a CMS tracker-hit ROOT ntuple and extracts hits matched to gen jets, writing two HDF5 tables: `hits` and `jets`.

**Key filters:**
- `--min-jet-pt`: minimum jet pT in GeV (default: 200)
- `--max-jet-eta`: maximum |η| (default: 2.0; use 1.0 for pure barrel)
- `--barrel-only`: keep only barrel sub-detectors (PixelBarrel, TIB, TOB)
- `--hit-types`: hit_type values to keep (default: `0 1`; drops stereo duplicates and exact-duplicate types 3/4)

**Usage:**
```bash
# QCD sample, barrel jets only
python3 extract_hit_jet_association.py 12220/qcd_test.root hits_jets_qcd.h5 \
    --jet-label QCD --max-jet-eta 1.0 --barrel-only

# ttbar sample
python3 extract_hit_jet_association.py 12220/ttbar_test.root hits_jets_ttbar.h5 \
    --jet-label ttbar --max-jet-eta 1.0 --barrel-only

# Process only the first 100 events (for testing)
python3 extract_hit_jet_association.py 12220/qcd_test.root hits_jets_qcd.h5 \
    --jet-label QCD --max-jet-eta 1.0 --barrel-only --max-events 100
```

**Output HDF5 tables:**

`hits` columns: `event_id`, `jet_id`, `hit_global_x/y/z`, `hit_local_x/y`, `hit_local_x/y_error`, `hit_sub_det`, `hit_layer`, `hit_type`

`jets` columns: `event_id`, `jet_id`, `jet_pt`, `jet_eta`, `jet_phi`, `n_hits`, `jet_label`

---

### 2. `analyze_hits_jets.py`

Single-file diagnostic plots for a hits_jets h5 file.

**Plots produced:**
- Jet kinematics: pT, η, φ distributions
- n_hits per jet distribution and n_hits vs pT scatter
- Hits per layer split by sub-detector
- Average hits per jet per global detector layer
- Average within-jet nearest-neighbour hit distance per layer
- Hit global x/y/z distributions and transverse (x-y) view

**Usage:**
```bash
python3 analyze_hits_jets.py hits_jets_ttbar.h5 --outdir plots_analyze
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_h5` | — | Path to hits_jets h5 file |
| `--outdir` | `plots` | Output directory for PNG plots |

---

### 3. `compare_hits_jets.py`

Overlaid comparison plots between two or more hits_jets h5 files (e.g. QCD vs ttbar). Produces both **all-jets** and **leading-jet-only** versions of every plot.

**Plots produced (each in `_all` and `_leading` variants):**
- Jet kinematics: pT, η, φ
- n_hits per jet distribution
- n_hits vs pT scatter
- Average hits per jet per global detector layer
- Average within-jet nearest-neighbour hit distance per layer

The leading jet is defined as the highest-pT jet in each event.

**Usage:**
```bash
python3 compare_hits_jets.py hits_jets_ttbar.h5 hits_jets_qcd.h5 \
    --labels ttbar QCD --outdir plots_compare
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_h5` | — | Two or more h5 files to compare |
| `--labels` | *(required)* | Label for each file, same order |
| `--outdir` | `plots_compare` | Output directory |

---

### 4. `jet_layer_intersections.py`

Computes where each jet axis crosses each cylindrical barrel layer and derives local hit coordinates relative to that crossing. Produces residual plots and per-layer jet shape visualisations.

**Key functions:**
- `jet_layer_intersection(eta, phi, R)` → `(x, y, z)` axis-crossing point
- `add_local_layer_coords(hits, jets)` → adds `local_arc = R·Δφ` and `local_dz = hit_z − R·sinh(η)` columns
- Leading jet all-layers plot: one panel per tracker layer showing hit scatter around jet axis
- Random jet all-layers plots: same for N randomly sampled jets

**Usage:**
```bash
# Basic: residual plots + jet shape on PixelBarrel layer 1 and TOB layer 6
python3 jet_layer_intersections.py hits_jets_ttbar.h5 hits_jets_qcd.h5 \
    --labels ttbar QCD --outdir plots_intersections

# Also plot 10 random jets per file
python3 jet_layer_intersections.py hits_jets_ttbar.h5 hits_jets_qcd.h5 \
    --labels ttbar QCD --outdir plots_intersections --random-jets 10 --seed 42

# Choose which layers to make jet-shape plots for (sub_det layer pairs)
python3 jet_layer_intersections.py hits_jets_ttbar.h5 \
    --labels ttbar --shape-layers 1 1 3 1 5 1 5 6
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_h5` | — | One or more h5 files |
| `--labels` | *(from filename)* | Label for each file |
| `--outdir` | `plots_intersections` | Output directory |
| `--max-jets` | `300` | Max jets used for residual scan (slow O(n·layers)) |
| `--shape-layers` | `1 1 5 6` | (sub_det, layer) pairs for jet-shape plots |
| `--random-jets` | `0` | Number of random jets to plot across all layers |
| `--seed` | `None` | Random seed for `--random-jets` sampling |

**Barrel layer radii used (empirical means from data):**

| Sub-detector | Layers | Radii [cm] |
|---|---|---|
| PixelBarrel | 1–3 | 4.38, 7.29, 10.16 |
| TIB | 1–4 | 25.52, 33.93, 41.76, 49.72 |
| TOB | 1–6 | 60.57, 69.39, 77.96, 86.75, 96.48, 107.99 |

---

### 5. `build_event_list.py`

Converts hits_jets tables into a sparse DVS-style `(t, x, y)` event list. Combines multiple files (QCD + ttbar) into one output.

**Coordinate encoding:**
- `t` = compact layer index 0–12 (barrel-only; endcap gaps removed)
- `x` = `local_arc` — arc-length offset from jet axis on that layer (cm)
- `y` = `local_dz` — z-offset from jet axis crossing on that layer (cm)

**Compact layer mapping (global_layer → timestep t):**
```
PixelBarrel layers 1-3  →  t = 0, 1, 2
TIB layers 1-4          →  t = 3, 4, 5, 6
TOB layers 1-6          →  t = 7, 8, 9, 10, 11, 12
```

**Usage:**
```bash
# Combine QCD and ttbar (labels read from each file's jet_label column)
python3 build_event_list.py hits_jets_qcd.h5 hits_jets_ttbar.h5 -o events.h5

# Override labels explicitly
python3 build_event_list.py hits_jets_qcd.h5 hits_jets_ttbar.h5 \
    --labels QCD ttbar -o events.h5
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_h5` | — | One or more hits_jets h5 files |
| `--labels` | *(from file)* | Override jet_label per file |
| `-o` / `--output` | `events.h5` | Output path |

**Output:** HDF5 with key `events`, columns: `event_id`, `jet_id`, `t`, `x`, `y`, `jet_label`, `jet_uid`

---

### 6. `bin_events_to_frames.py`

Bins the sparse event list into dense per-jet 2D histogram frames: one histogram per tracker layer, stacked into shape `(N_jets, T=13, H, W)`.

**Binning strategy (variable-resolution):**
Each layer's physical extent (in cm) grows with radius. Rather than using a fixed bin count per layer (which makes outer-layer pixels enormous), a fixed physical pixel size is used so outer layers automatically get more, finer bins. All layers are then zero-padded into a shared canvas of `max_bins × max_bins`.

With `--pixel-size 10 --max-bins 64` the bin counts are approximately:
```
t=0 (PixelBarrel l1): 6 bins    t=7  (TOB l1): 37 bins
t=1 (PixelBarrel l2): 6 bins    t=8  (TOB l2): 43 bins
t=2 (PixelBarrel l3): 7 bins    t=9  (TOB l3): 48 bins
t=3 (TIB l1):        16 bins    t=10 (TOB l4): 53 bins
t=4 (TIB l2):        21 bins    t=11 (TOB l5): 58 bins
t=5 (TIB l3):        26 bins    t=12 (TOB l6): 64 bins
t=6 (TIB l4):        31 bins
```

**Usage:**
```bash
# Variable-resolution (recommended)
python3 bin_events_to_frames.py events.h5 -o frames.npz --pixel-size 10 --max-bins 64

# Fixed-resolution baseline (every layer gets the same 32×32 grid)
python3 bin_events_to_frames.py events.h5 -o frames_fixed32.npz --pixel-size 999 --max-bins 32
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_h5` | — | events.h5 from `build_event_list.py` |
| `-o` / `--output` | `frames.npz` | Output path |
| `--pixel-size` | `1.5` | Physical pixel size in cm |
| `--max-bins` | `64` | Cap on bins per axis per layer |
| `--min-bins` | `4` | Floor on bins per axis per layer |
| `--percentile` | `99.0` | Percentile used to set each layer's bin range |

**Output `frames.npz` keys:** `frames` (N, T, H, W), `labels` (N,), `t_list` (T,), `jet_uids` (N,)

---

### 7. `train_snn.py`

Trains a spiking CNN (snnTorch) on the frame tensor for binary QCD vs ttbar classification.

**Architecture (`JetSNN`):**
```
Input: (batch, T=13, 1, H, W) — binarized frames
  Conv2d(1→8, k=5) → LIF → MaxPool(2)
  Conv2d(8→16, k=5) → LIF → MaxPool(2) → MaxPool(2)
  Linear(flat→32) → LIF
  Linear(32→2) → LIF
Output: summed spike counts over T timesteps (rate-coded logits)
```

- ~22k parameters for 64×64 input
- Frames binarized before training: pixel = 1 if any hit fell in that bin, else 0
- Membrane state carried across all 13 timesteps per forward pass
- Architecture uses only standard `Conv2d`/`Linear` + fixed-parameter `snn.Leaky` neurons for hls4ml portability

**Usage:**
```bash
python3 train_snn.py frames.npz --epochs 30

# Full options
python3 train_snn.py frames.npz --epochs 50 --batch-size 64 --lr 5e-4 --val-frac 0.2 --seed 0
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `input_npz` | — | frames.npz from `bin_events_to_frames.py` |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `32` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--val-frac` | `0.2` | Fraction of jets held out for validation |
| `--seed` | `0` | Random seed for reproducibility |

---

## Full Pipeline Example

```bash
# 1. Extract hits and jets from ROOT files
python3 extract_hit_jet_association.py 12220/qcd_test.root hits_jets_qcd.h5 \
    --jet-label QCD --max-jet-eta 1.0 --barrel-only

python3 extract_hit_jet_association.py 12220/ttbar_test.root hits_jets_ttbar.h5 \
    --jet-label ttbar --max-jet-eta 1.0 --barrel-only

# 2. (Optional) Diagnostic plots
python3 analyze_hits_jets.py hits_jets_ttbar.h5 --outdir plots_analyze
python3 compare_hits_jets.py hits_jets_ttbar.h5 hits_jets_qcd.h5 \
    --labels ttbar QCD --outdir plots_compare
python3 jet_layer_intersections.py hits_jets_ttbar.h5 hits_jets_qcd.h5 \
    --labels ttbar QCD --outdir plots_intersections --random-jets 5 --seed 0

# 3. Build sparse event list
python3 build_event_list.py hits_jets_qcd.h5 hits_jets_ttbar.h5 -o events.h5

# 4. Bin into frames
python3 bin_events_to_frames.py events.h5 -o frames.npz --pixel-size 10 --max-bins 64

# 5. Train SNN
python3 train_snn.py frames.npz --epochs 30
```

---

## Data

Input ROOT files come from [CERN Open Data record 12220](https://opendata.cern.ch/record/12220):
- `qcd_test.root` — QCD multijet sample, ~768 MB
- `ttbar_test.root` — ttbar sample, ~960 MB

Download with [cernopendata-client](https://cernopendata-client.readthedocs.io/):
```bash
cernopendata-client download-files --recid 12220 --filter-range 1-1  # first file
```

**Scale note:** The two test files yield ~1,987 jets total (~755 QCD + ~1,232 ttbar) with `|η| < 1.0`, `pT > 200 GeV`. Meaningful training requires ~200k jets. Additional files from the same record or produced via [record 12210](https://opendata.cern.ch/record/12210) (TrackerRecHitProducerTool) are needed to reach that scale.
