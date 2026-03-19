# DAG-deepVASE RNA — Causal Gene Regulatory Network Pipeline

A genome-wide causal network inference pipeline for bulk RNA-seq data, built on top of [DAG-deepVASE](https://github.com/ZhenjiangFan/DAG-deepVASE). It combines MB-LASSO skeleton learning, DNN knockoff selection, STRING PPI filtering, and Degenerate Gaussian Score edge orientation to produce a directed acyclic gene regulatory network from expression data.

---

## Table of Contents

- [Overview](#overview)
- [Pipeline Steps](#pipeline-steps)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration — Read This First](#configuration--read-this-first)
- [Running the Pipeline](#running-the-pipeline)
- [Outputs](#outputs)
- [Hardware Notes](#hardware-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

```
Raw TSV files (per sample)
        ↓
build_rnaseq_matrix.py   — builds log1p-TPM gene × sample matrix
        ↓
causal_network_global_1sample.py
  Stage 1 : Knockoff matrix generation   (cached)
  Stage 2 : MB-LASSO skeleton            (cached)
  Stage 3 : STRING PPI filter
  Stage 4 : DNN knockoff selection       (checkpointed)
  Stage 5 : Three-way intersection       MB-LASSO ∩ DNN ∩ PPI
  Stage 6 : Edge orientation             Degenerate Gaussian Score
  Stage 7 : DAG enforcement              weakest-edge cycle removal
  Stage 8 : Save                         CSV with regulation sign
```

---

## Pipeline Steps

### Script 1 — `build_rnaseq_matrix.py`

Reads per-sample TSV files (ENCODE format or similar), strips Ensembl version suffixes, maps IDs to HGNC gene symbols via MyGene.info, filters low-expression genes, and writes two matrices: a full log1p-TPM matrix and a top-N variance-filtered version for causal analysis.

### Script 2 — `causal_network_global_1sample.py`

Takes the causal matrix and runs the full DAG-deepVASE pipeline, producing a directed, acyclic gene regulatory network with edges labelled as activation or repression.

---

## Requirements

### Operating system

Linux (Ubuntu 20.04+ recommended). 

### Python

Python 3.9 or 3.10 recommended. Python 3.11+ has not been tested with all dependencies.

### Core Python packages

```
pandas >= 1.5
numpy >= 1.24
mygene
tqdm
joblib
scikit-learn
networkx
scipy
tensorflow >= 2.12
```

Optional (for GPU-accelerated inverse covariance):
```
cupy-cuda11x   # CUDA 11.x
cupy-cuda12x   # CUDA 12.x
```

---

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/ipranjalisharma/dag-deepvase-rna.git
cd dag-deepvase-rna
```

### 2. Clone the DAG-deepVASE repository

```bash
git clone https://github.com/ZhenjiangFan/DAG-deepVASE
```

Note the path where you cloned it — you will need to set `DAG_DEEPVASE_PATH` in the config.

### 3. Create a conda environment

```bash
conda create -n causal_env python=3.10
conda activate causal_env
```

### 4. Install Python dependencies

#### CPU-only installation

```bash
pip install pandas numpy mygene tqdm joblib scikit-learn networkx scipy tensorflow
```

#### GPU installation (recommended for the DNN stage)

First check your CUDA version:

```bash
nvidia-smi
```

Then install the matching TensorFlow and CuPy:

**CUDA 11.x:**
```bash
pip install tensorflow[and-cuda]==2.13.*
pip install cupy-cuda11x
```

**CUDA 12.x:**
```bash
pip install tensorflow[and-cuda]==2.15.*
pip install cupy-cuda12x
```

Verify GPU is visible to TensorFlow:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see output like `[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]`. If the list is empty, see [Troubleshooting](#troubleshooting).

---

## Configuration — Read This First

> **Both scripts have a clearly marked `CONFIGURATION` block near the top of the file. You must edit these blocks before running.**

The scripts will raise an error and exit immediately if any placeholder path is detected.

---

### `build_rnaseq_matrix.py` — Configuration block

Open the file and set the following:

| Variable | What to set |
|---|---|
| `DATA_ROOT` | Path to the folder containing one sub-folder per sample, each with one `.tsv` file |
| `OUTPUT_FILE` | Full path (including filename) for the output matrix |
| `EXPRESSION_COL` | Column name in your TSV files — typically `"TPM"`, `"FPKM"`, or `"expected_count"` |
| `MAX_ZERO_FRACTION` | Drop genes that are zero in more than this fraction of samples. Default `0.8` |
| `LOG_TRANSFORM` | `True` recommended for all downstream analysis |
| `TOP_CAUSAL_GENES` | Number of top-variance genes to keep for causal input. Default `5000` |

Expected folder layout for `DATA_ROOT`:

```
data/
  K562/
    ENCFF001.tsv        <- must contain 'gene_id' and your EXPRESSION_COL
  HepG2/
    ENCFF002.tsv
  MCF7/
    ENCFF003.tsv
```

---

### `causal_network_global_1sample.py` — Configuration block

#### Paths

| Variable | What to set |
|---|---|
| `DATA_FOLDER` | Project folder — all intermediate and output files go here |
| `MATRIX_FILE` | Filename of the causal input matrix (the `_topN_causal.txt` file from Step 1) |
| `KNOCKOFF_FILE` | Filename for the knockoff matrix cache (generated automatically) |
| `MB_SKELETON_FILE` | Filename for the MB-LASSO skeleton cache (generated automatically) |
| `OUTPUT_FILE` | Filename for the final network CSV |
| `TMP_DIR` | Directory for per-gene DNN temp files. `/tmp` is fine; use a fast local SSD if `/tmp` is small or RAM-backed |
| `PPI_FILE` | Path to the STRING PPI file. Set to `None` to skip PPI filtering |
| `DAG_DEEPVASE_PATH` | Path to the cloned `dag-deepvase` repository |

#### GPU / CPU settings

> **You must check your hardware and set these values accordingly.**

| Variable | How to set it |
|---|---|
| `CUDA_VISIBLE_DEVICES` | Run `nvidia-smi` in a terminal. Use `"0"` for the first GPU, `"0,1"` for two GPUs, `""` to force CPU-only |
| `XLA_CUDA_DIR` | Only needed if CUDA was installed via pip (nvidia-* packages). Find the path with the command below. Set to `None` if CUDA was installed system-wide (apt or conda) |

To find `XLA_CUDA_DIR` if needed:

```bash
python -c "import nvidia.cuda_nvcc, os; print(os.path.dirname(nvidia.cuda_nvcc.__file__))"
```

Set `XLA_CUDA_DIR` to the path printed.

#### Algorithm settings

These have sensible defaults but should be reviewed for your dataset:

| Variable | Default | Notes |
|---|---|---|
| `PPI_SCORE_THRESHOLD` | `400` | STRING confidence score 0-1000. Use `700` for high-confidence only |
| `MB_ALPHA` | `0.1` | LASSO regularisation. Use `0.1` for n < 500, `0.05` for n > 1000 |
| `DNN_EPOCHS` | `20` | Increase to `30-50` if convergence is poor |
| `DNN_BATCH_SIZE` | `32` | Rule of thumb: n / 15. For n=400: `32`. For n=2000: `128` |
| `FDR_Q` | `0.05` | Target false discovery rate. Lower = fewer, more confident edges |

---

## Running the Pipeline

### Step 1 — Build the expression matrix

```bash
conda activate causal_env
python build_rnaseq_matrix.py
```

This produces two files:
- `RNAseq_Matrix.txt` — full matrix
- `RNAseq_Matrix_top5000_causal.txt` — variance-filtered input for Step 2

Set `MATRIX_FILE` in `causal_network_global_1sample.py` to the `_topN_causal.txt` filename before continuing.

### Step 2 — Run the causal network pipeline

```bash
python causal_network_global_1sample.py
```

The pipeline is **checkpointed** — if it is interrupted during the DNN stage (Stage 4), re-running the script will resume from where it left off rather than restarting. Stages 1 and 2 are also cached to disk and skipped on reruns.

To force a full rerun, delete the cache files from `DATA_FOLDER`:

```bash
rm Knockoff_Matrix.txt MB_Skeleton.csv dnn_checkpoint.json
rm -rf DNNSelection/
```

---

## Outputs

| File | Location | Description |
|---|---|---|
| `RNAseq_Matrix.txt` | `OUTPUT_FILE` dir | Full log1p-TPM matrix, all genes |
| `RNAseq_Matrix_top5000_causal.txt` | Same dir | Top-variance genes for causal input |
| `Knockoff_Matrix.txt` | `DATA_FOLDER` | LedoitWolf knockoff matrix (Stage 1 cache) |
| `MB_Skeleton.csv` | `DATA_FOLDER` | MB-LASSO skeleton edges (Stage 2 cache) |
| `DNNSelection/` | `DATA_FOLDER` | Per-gene DNN selection CSVs (Stage 4 cache) |
| `Global_Causal_Network.csv` | `DATA_FOLDER` | **Final output** — directed causal edges |

### Network CSV columns

| Column | Description |
|---|---|
| `Cause` | Regulator gene (source node) |
| `Effect` | Target gene (destination node) |
| `EffectSize` | log(DG Score difference) — larger = stronger evidence for direction |
| `Regulation` | `Positive (Activation)` or `Negative (Repression)` based on partial correlation |

---

## Hardware Notes

> **These settings need to be customised for your machine. Do not run with defaults if your hardware differs from the examples below.**

### GPU

The DNN stage (Stage 4) benefits significantly from a GPU. On a dataset with ~5000 genes and ~400 samples, typical runtimes are:

| Hardware | Approximate DNN stage time |
|---|---|
| NVIDIA RTX 3090 / A100 | 2-6 hours |
| NVIDIA RTX 3060 / T4 | 6-12 hours |
| CPU only (16 cores) | 24-72+ hours |

Set `CUDA_VISIBLE_DEVICES = ""` to force CPU-only if no GPU is available. The pipeline will print `[GPU] No GPU detected — running on CPU` at startup to confirm.

CuPy (optional) accelerates Stage 6 (inverse covariance). For large gene sets (>5000 genes), this saves meaningful time. Install with `pip install cupy-cuda11x` or `cupy-cuda12x` as appropriate.

### CPU cores

CPU core count is **detected automatically** via `multiprocessing.cpu_count()`. All available logical cores are used for the MB-LASSO stage (capped at 64 to prevent RAM saturation). No manual setting is needed.

If you want to limit CPU usage (e.g. on a shared machine), find and change this line in `causal_network_global_1sample.py`:

```python
# Default — uses all logical cores
N_CORES = multiprocessing.cpu_count()

# Override — change to however many cores you want to allow
N_CORES = 8
```

### RAM

- Stage 1 (knockoffs): needs roughly `2 x n x p x 4 bytes` of RAM. For 5000 genes x 400 samples: ~16 GB.
- Stage 2 (MB-LASSO): uses memory-mapped files to reduce per-worker RAM usage.
- Stage 4 (DNN): loads the full knockoff matrix into RAM (~same as Stage 1). GPU VRAM usage is typically under 4 GB per gene batch.

---

## Troubleshooting

**`[ERROR] DATA_FOLDER has not been set`**
The placeholder paths have not been replaced. Edit the `CONFIGURATION` block at the top of the relevant script.

**`Could not load dynamic library 'libcuda.so.1'`**
TensorFlow cannot find the CUDA libraries. Try:
```bash
export LD_LIBRARY_PATH=$(python -c "import site, os; d=os.path.join(site.getsitepackages()[0],'nvidia'); print(':'.join([os.path.join(d,x,'lib') for x in os.listdir(d) if os.path.exists(os.path.join(d,x,'lib'))]))"):$LD_LIBRARY_PATH
```
Or set `XLA_CUDA_DIR` in the configuration block.

**`No GPU detected — running on CPU`**
Check that the correct CUDA version is installed, then run:
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
If the list is empty, reinstall TensorFlow: `pip install tensorflow[and-cuda]`

**`Knockoff Cholesky decomposition failed`**
The covariance matrix is not positive definite after LedoitWolf shrinkage. Try reducing the number of input genes (lower `TOP_CAUSAL_GENES`) or check for near-zero-variance genes in your matrix.

**`No edges survived intersection`**
The three-way filter is too strict for your data. Try one or more of:
- Lower `MB_ALPHA` from `0.1` to `0.05`
- Lower `PPI_SCORE_THRESHOLD` from `400` to `200`
- Raise `FDR_Q` from `0.05` to `0.1`
- Set `PPI_FILE = None` to skip the PPI filter entirely

**DNN stage is very slow on CPU**
This is expected. Each gene requires a small neural network training run. With 5000 genes, CPU runtime can exceed 48 hours. A GPU is strongly recommended.

**`mygene` returns 0 mapped genes**
Ensure your `gene_id` column contains Ensembl IDs (starting with `ENSG`). The script strips version suffixes automatically. Check the format with:
```bash
head -5 your_sample.tsv
```

---

## Citation

If you use this pipeline, please cite the original DAG-deepVASE paper and the underlying tools:

- **DAG-deepVASE** — Orenstein Y. et al. *DAG-deepVASE: A Deep Learning Approach for Variable Selection and Causal Structure Learning* (please cite the original repository)
- **Knockoff filter** — Candes E. et al. (2018). *Panning for Gold: Model-X Knockoffs for High-Dimensional Controlled Variable Selection.* JRSS-B, 80(3).
- **LedoitWolf** — Ledoit O. & Wolf M. (2004). *A well-conditioned estimator for large-dimensional covariance matrices.* J. Multivariate Analysis, 88(2).
- **STRING** — Szklarczyk D. et al. (2023). *The STRING database in 2023.* Nucleic Acids Research, 51(D1).
- **mygene** — Wu C. et al. (2013). *BioGPS and MyGene.info: organizing online, gene-centric information.* Nucleic Acids Research, 41(D1).
