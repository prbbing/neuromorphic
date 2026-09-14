# CERN Open Data — Portal & Client Guide

## Overview

The [CERN Open Data Portal](https://opendata.cern.ch) provides public access to collision data, simulated events, software, and documentation from LHC experiments (ATLAS, CMS, LHCb, ALICE).

Data is organised by **record ID**. Each record has metadata, a list of files, and a DOI for citation.

---

## 1. Finding Data on the Portal

### Browse by experiment / dataset type

- Go to https://opendata.cern.ch
- Use the left-hand filters: **Experiment**, **Type** (Dataset, Software, Workflow), **Year**, **Energy**

### Search for a specific dataset

Use the search bar or append a query to the URL:

```
https://opendata.cern.ch/search?q=ttbar+simulation+ATLAS
```

### Record page

Each record has:
- **Record ID** (e.g. `15005`) — used by the client
- File list with sizes and checksums
- README / documentation
- DOI for citation (cite this in papers)

---

## 2. Installing the CERN Open Data Client

```bash
pip install cernopendata-client
```

Verify:

```bash
cernopendata-client --version
```

The client wraps the REST API and handles file listing, download, and checksum verification.

---

## 3. CLI Quick Reference

### List files in a record

```bash
cernopendata-client get-file-locations --recid 15005
```

This prints the logical file names (LFNs) and their sizes without downloading anything.

### Download a specific file

```bash
cernopendata-client download-files --recid 15005 \
    --filter-name "mc-flavtag-ttbar-small.h5"
```

### Download all files in a record

```bash
cernopendata-client download-files --recid 15005
```

Files land in the current directory by default. Use `--output-dir` to specify a path:

```bash
cernopendata-client download-files --recid 15005 \
    --output-dir /data/cern/
```

### Verify checksums after download

```bash
cernopendata-client verify-files --recid 15005
```

### List fields available in a record's metadata

```bash
cernopendata-client get-metadata --recid 15005
```

### Get a single metadata field (e.g. title)

```bash
cernopendata-client get-metadata --recid 15005 --output-field title
```

---

## 4. Python API

The client also exposes a Python interface:

```python
from cernopendata_client import search_records, get_file_locations, download_files

# Search for records
records = search_records(query="ATLAS ttbar open data")
for r in records:
    print(r["id"], r["metadata"]["title"])

# List files in a record
files = get_file_locations(recid=15005)
for f in files:
    print(f["uri"], f["size"])

# Download
download_files(recid=15005, output_dir="./data/")
```

---

## 5. REST API (curl / requests)

You can also query the portal directly without the client:

### Record metadata

```bash
curl "https://opendata.cern.ch/api/records/15005" | python3 -m json.tool
```

### File list from metadata

```python
import requests

r = requests.get("https://opendata.cern.ch/api/records/15005").json()
for f in r["metadata"]["files"]:
    print(f["key"], f["size"])
    # direct download URL:
    print(f"https://opendata.cern.ch/record/15005/files/{f['key']}")
```

### Download a file directly

```bash
curl -O "https://opendata.cern.ch/record/15005/files/mc-flavtag-ttbar-small.h5"
```

---

## 6. ATLAS Flavour-Tagging Open Data

The dataset used in this project is the **ATLAS flavour-tagging open data** released alongside the SALT framework.

| Item | Detail |
|---|---|
| Portal record | https://opendata.cern.ch/record/15005 |
| File | `mc-flavtag-ttbar-small.h5` |
| Process | t t̄ MC simulation, √s = 13 TeV |
| Jets | ~5.6 M anti-kT R=0.4 jets |
| Format | HDF5 with structured arrays: `jets`, `tracks`, `truth_hadrons`, `eventwise` |

Download command:

```bash
cernopendata-client download-files --recid 15005 \
    --filter-name "mc-flavtag-ttbar-small.h5" \
    --output-dir ./
```

---

## 7. XRootD (for large files / batch jobs)

Files are also accessible via XRootD for direct streaming without a full download:

```bash
xrdcp "root://eospublic.cern.ch//eos/opendata/atlas/..." ./local_copy.root
```

The XRootD URI for each file appears in the portal's file list under **"root://"** links.  
Install XRootD: `conda install -c conda-forge xrootd` or `pip install xrootd`.

For HDF5 files, full download is usually preferable over streaming.

---

## 8. Citation

Always cite the dataset DOI in papers and notes. The DOI is on the record page, e.g.:

```
ATLAS Collaboration (2024). ATLAS flavour-tagging open data.
CERN Open Data Portal. DOI: 10.7483/OPENDATA.ATLAS.XXXXX
```

---

## 9. Useful Links

| Resource | URL |
|---|---|
| Portal | https://opendata.cern.ch |
| Client docs | https://cernopendata-client.readthedocs.io |
| Client source | https://github.com/cernopendata/cernopendata-client |
| SALT framework | https://github.com/umami-hep/salt |
| ATLAS open data overview | https://opendata.cern.ch/search?experiment=ATLAS |
