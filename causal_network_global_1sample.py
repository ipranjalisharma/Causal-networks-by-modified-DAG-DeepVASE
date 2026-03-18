#!/usr/bin/env python3
"""
Global Causal Network Pipeline  (DAG-deepVASE adapted)

Scale-adapted DAG-deepVASE for genome-wide RNA-seq data.

Stages:
  1. Knockoff matrix generation    (once, cached)
  2. MB-LASSO skeleton             (once, cached)
  3. STRING PPI filter             (restricts DNN to PPI-covered genes)
  4. DNN knockoff selection        (skeleton neighbours as inputs, checkpointed)
  5. Three-way intersection        MB-LASSO ∩ DNN ∩ PPI
  6. Edge orientation              Degenerate Gaussian Score
  7. DAG enforcement               remove weakest cycle edge iteratively
  8. Save                          CSV with partial-correlation regulation sign

For n ~ 400 samples:
  - LedoitWolf shrinkage for knockoff covariance (sample cov is singular when n < p)
  - MB_ALPHA raised to 0.1 (less strict than 0.05 — gives more edges at low n)
  - DNN_BATCH_SIZE 32  (fits ~400 samples cleanly; ~13 batches/epoch)

Checkpointed — safe to Ctrl+C and resume at any point during the DNN stage.

-----------------------------------------------
 USER CONFIGURATION — edit the block below

"""

import os
import sys
import multiprocessing

#  CONFIGURATION  ← Edit these values before running


#Paths 

# Folder where all data files live and where output files will be written.
DATA_FOLDER = "/path/to/your/project/folder"

# Expression matrix filename (inside DATA_FOLDER).
# Tab-separated; genes as rows, samples as columns.
# This is the _top5k_causal.txt file produced by build_rnaseq_matrix.py.
MATRIX_FILE = "RNAseq_Matrix_top5000_causal.txt"

# Filenames for cached intermediate results (generated automatically on first run).
# Delete these files to force a re-run of the corresponding stage.
KNOCKOFF_FILE    = "Knockoff_Matrix.txt"
MB_SKELETON_FILE = "MB_Skeleton.csv"
OUTPUT_FILE      = "Global_Causal_Network.csv"

# Temporary directory used during DNN training per gene.
# /tmp is fine on Linux; change to a fast local SSD path if /tmp is small.
# Example: TMP_DIR = "/scratch/your_username/dnn_tmp"
TMP_DIR = "/tmp"

# STRING PPI file (tab-separated, gene_a / gene_b / score columns).
# Set to None to skip the PPI filter and run a two-stage pipeline only.
# Generate with:  python prepare_string_ppi.py  (see README)
PPI_FILE = os.path.join(DATA_FOLDER, "PPI", "STRING_PPI_filtered.txt")

# Path to the cloned DAG-deepVASE repository.
# git clone https://github.com/orzuk/dag-deepvase.git
DAG_DEEPVASE_PATH = "/path/to/dag-deepvase"

#  GPU / CPU settings 
# Which GPU(s) to use.  "0" = first GPU, "0,1" = first two, "" = CPU only.
# Run  nvidia-smi  in a terminal to see available GPUs and their indices.
CUDA_VISIBLE_DEVICES = "0"

# Path to the CUDA nvcc binary directory used by XLA (TensorFlow JIT compiler).
# Only needed if you installed CUDA via pip (nvidia-* packages).
# Find it with:
#   python -c "import nvidia.cuda_nvcc; print(nvidia.cuda_nvcc.__file__)"
# then take the directory containing 'bin/nvcc'.
# Set to None to skip (safe if you installed CUDA system-wide via apt/conda).
XLA_CUDA_DIR = None   # e.g. "/home/user/miniconda3/envs/causal_env/lib/python3.10/site-packages/nvidia/cuda_nvcc"

# Algorithm settings 

# STRING PPI confidence score threshold (0–1000).
# 400 = medium confidence. Raise to 700 for high-confidence edges only.
PPI_SCORE_THRESHOLD = 400

# MB-LASSO regularisation strength.
# 0.1 is recommended for n ~ 400 samples (lenient enough to capture weak edges).
# Lower to 0.05 if you have n > 1000 samples.
MB_ALPHA = 0.1

# DNN training settings.
DNN_EPOCHS     = 20
DNN_BATCH_SIZE = 32   # Rule of thumb: ~n/15.  For n=400: 32.  For n=2000: 128.

# FDR (false discovery rate) control.
FDR_Q      = 0.05   # Target FDR level.
FDR_OFFSET = 1      # 1 = conservative (knockoff+), 0 = standard knockoff.

# Fraction of unique values below which a variable is treated as discrete
# for DG Score edge orientation. 0.2 is a safe default.
DISCRETE_THRESHOLD = 0.2



#  Apply GPU / environment settings 

if XLA_CUDA_DIR:
    os.environ["XLA_FLAGS"] = f"--xla_gpu_cuda_data_dir={XLA_CUDA_DIR}"

os.environ["CUDA_VISIBLE_DEVICES"]  = CUDA_VISIBLE_DEVICES
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "2"
os.environ["TF_GPU_THREAD_MODE"]    = "gpu_private"

import site

N_CORES = multiprocessing.cpu_count()


def _fix_ld_path():
    """Add nvidia library paths so TF can find pip-installed CUDA libs."""
    nvidia_dir = os.path.join(site.getsitepackages()[0], "nvidia")
    if not os.path.exists(nvidia_dir):
        return
    paths = [
        os.path.join(nvidia_dir, d, "lib")
        for d in os.listdir(nvidia_dir)
        if os.path.exists(os.path.join(nvidia_dir, d, "lib"))
    ]
    if paths:
        existing = os.environ.get("LD_LIBRARY_PATH", "")
        os.environ["LD_LIBRARY_PATH"] = ":".join(paths) + (":" + existing if existing else "")


_fix_ld_path()

os.environ["OMP_NUM_THREADS"]      = str(N_CORES)
os.environ["OPENBLAS_NUM_THREADS"] = str(N_CORES)
os.environ["MKL_NUM_THREADS"]      = str(N_CORES)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"[GPU] {len(gpus)} GPU(s) detected — memory growth enabled")
else:
    print("[GPU] No GPU detected — running on CPU (DNN stage will be slow)")
print(f"[CPU] {N_CORES} logical core(s) available")

import json
import shutil
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx
import scipy.stats
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.linear_model import Lasso
from sklearn.covariance import LedoitWolf

try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("[GPU] CuPy available — inverse covariance will run on GPU")
except ImportError:
    CUPY_AVAILABLE = False

CHECKPOINT_FILE = os.path.join(DATA_FOLDER, "dnn_checkpoint.json")

sys.path.insert(0, DAG_DEEPVASE_PATH)


#  Startup validation 

def _validate_config():
    """Fail early with a clear message if placeholder paths have not been set."""
    errors = []
    if DATA_FOLDER == "/path/to/your/project/folder":
        errors.append("  DATA_FOLDER has not been set.")
    if DAG_DEEPVASE_PATH == "/path/to/dag-deepvase":
        errors.append("  DAG_DEEPVASE_PATH has not been set.")
    if not os.path.isdir(DATA_FOLDER):
        errors.append(f"  DATA_FOLDER does not exist: {DATA_FOLDER}")
    if not os.path.isdir(DAG_DEEPVASE_PATH):
        errors.append(f"  DAG_DEEPVASE_PATH does not exist: {DAG_DEEPVASE_PATH}")
    if errors:
        raise SystemExit(
            "\n[ERROR] Configuration issues found:\n" + "\n".join(errors) +
            "\n\nOpen causal_network_global_1sample.py and edit the CONFIGURATION block at the top."
        )


#  Checkpoint helpers 

def load_checkpoint() -> set:
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            done = set(json.load(f).get("completed_genes", []))
        print(f"  [Checkpoint] Resuming — {len(done):,} genes already done.")
        return done
    return set()


def save_checkpoint(completed: set):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"completed_genes": list(completed)}, f)


def clear_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


# Stage 0: Load expression matrix 

def load_dataset() -> pd.DataFrame:
    """
    Load expression matrix and transpose to samples × genes.
    Expects genes as rows, samples as columns on disk.
    """
    path = os.path.join(DATA_FOLDER, MATRIX_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Matrix file not found: {path}\n"
            f"Run build_rnaseq_matrix.py first, or check MATRIX_FILE in the config."
        )
    print(f"\nLoading: {path}")
    df = pd.read_csv(path, sep="\t", index_col=0)
    print(f"  On disk    : {df.shape[0]:,} genes × {df.shape[1]:,} samples")
    df = df.T
    print(f"  Transposed : {df.shape[0]:,} samples × {df.shape[1]:,} genes")
    return df


#  Stage 1: Knockoff generation 

def generate_knockoffs(dataset: pd.DataFrame) -> None:
    """
    Generate second-order Gaussian knockoffs using LedoitWolf shrinkage.

    LedoitWolf is used instead of sample covariance because when n < p
    (e.g. 410 samples, 28k genes), the sample covariance matrix is
    rank-deficient and Cholesky decomposition fails. LedoitWolf produces a
    regularised shrinkage estimate that is always positive definite.
    """
    ko_path = os.path.join(DATA_FOLDER, KNOCKOFF_FILE)
    if os.path.exists(ko_path):
        print(f"\n  Knockoff matrix already exists: {ko_path}")
        print(f"  Delete it to regenerate.")
        return

    print("\n[Stage 1] Generating knockoff matrix...")
    print(f"  n={dataset.shape[0]}, p={dataset.shape[1]} — using LedoitWolf shrinkage")

    X    = dataset.values.astype(np.float64)
    n, p = X.shape

    t0 = time.time()
    print("  Fitting LedoitWolf covariance...")
    lw = LedoitWolf(assume_centered=False)
    lw.fit(X)
    Sigma = lw.covariance_
    print(f"  Covariance fit: {time.time()-t0:.1f}s")

    t1 = time.time()
    print("  Cholesky decomposition...")
    L = np.linalg.cholesky(Sigma + np.eye(p) * 1e-6)
    print(f"  Cholesky: {time.time()-t1:.1f}s")

    t2 = time.time()
    print("  Generating knockoff samples...")
    s_val = min(2 * np.linalg.eigvalsh(Sigma).min(), 1.0)
    s_val = max(s_val, 1e-4)
    S     = np.diag(np.full(p, s_val))

    Sigma_inv   = np.linalg.pinv(Sigma)
    Sigma_tilde = 2 * S - S @ Sigma_inv @ S
    Sigma_tilde += np.eye(p) * 1e-6
    L_tilde = np.linalg.cholesky(Sigma_tilde)

    mu   = X.mean(axis=0)
    X_c  = X - mu
    X_ko = (X_c - X_c @ Sigma_inv @ S + np.random.randn(n, p) @ L_tilde.T) + mu
    print(f"  Knockoff construction: {time.time()-t2:.1f}s")

    combined  = np.hstack([X, X_ko]).astype(np.float32)
    col_names = dataset.columns.tolist()
    ko_cols   = [f"{c}_knockoff" for c in col_names]
    df_ko     = pd.DataFrame(combined, index=dataset.index,
                             columns=col_names + ko_cols)
    print(f"  Saving knockoff matrix ({combined.nbytes/1e9:.2f} GB)...")
    df_ko.to_csv(ko_path, sep="\t")
    print(f"  Saved: {ko_path}  ({time.time()-t0:.1f}s total)")


#  Stage 2: MB-LASSO skeleton 

def _lasso_one_gene(j: int, mmap_path: str, alpha: float, col_names: list) -> list:
    """
    Fit LASSO for gene j using all other genes as predictors.
    Reads from a memory-mapped .npy file so parallel workers share the matrix
    without each holding a full RAM copy.
    """
    X    = np.load(mmap_path, mmap_mode="r")
    y    = X[:, j].copy()
    keep = list(range(j)) + list(range(j + 1, X.shape[1]))
    Xj   = X[:, keep]
    cols = [col_names[i] for i in keep]
    try:
        model = Lasso(alpha=alpha, max_iter=10000)
        model.fit(Xj, y)
        return [cols[i] for i, c in enumerate(model.coef_) if c != 0.0]
    except Exception:
        return []


def build_mb_skeleton(dataset: pd.DataFrame) -> set:
    """
    Markov Blanket skeleton via LASSO with AND rule.
    Edge (i,j) is kept only if both gene i selects j AND gene j selects i.
    Cached to MB_SKELETON_FILE — delete to recompute.
    """
    skeleton_path = os.path.join(DATA_FOLDER, MB_SKELETON_FILE)

    if os.path.exists(skeleton_path):
        print(f"\n  Loading cached skeleton: {skeleton_path}")
        df  = pd.read_csv(skeleton_path)
        out = set(
            "___".join(sorted([r["Feature1"], r["Feature2"]]))
            for _, r in df.iterrows()
        )
        print(f"  Skeleton edges: {len(out):,}")
        return out

    print(f"\n[Stage 2] MB-LASSO skeleton")
    print(f"  alpha={MB_ALPHA}  |  AND rule  |  {N_CORES} cores")
    print(f"  n={dataset.shape[0]}, p={dataset.shape[1]}")

    col_names = dataset.columns.tolist()
    p         = len(col_names)
    X         = dataset.values.astype(np.float32)
    X_std     = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    mmap_path = os.path.join(DATA_FOLDER, "_mb_mmap.npy")
    np.save(mmap_path, X_std)

    # Cap workers to prevent RAM saturation from parallel matrix slicing
    safe_cores = min(N_CORES, 64)
    print(f"  Using {safe_cores} parallel workers")

    t0 = time.time()
    neighbour_lists = Parallel(n_jobs=safe_cores, verbose=0)(
        delayed(_lasso_one_gene)(j, mmap_path, MB_ALPHA, col_names)
        for j in tqdm(range(p), desc="  MB-LASSO", unit="gene")
    )

    if os.path.exists(mmap_path):
        os.remove(mmap_path)

    nbr = {col_names[j]: set(neighbour_lists[j]) for j in range(p)}

    skeleton = set()
    rows     = []
    for j in range(p):
        gene_j = col_names[j]
        for gene_k in nbr[gene_j]:
            if gene_j in nbr.get(gene_k, set()):
                key = "___".join(sorted([gene_j, gene_k]))
                if key not in skeleton:
                    skeleton.add(key)
                    rows.append({"Feature1": gene_j, "Feature2": gene_k})

    pd.DataFrame(rows).to_csv(skeleton_path, index=False)
    print(f"  Skeleton edges: {len(skeleton):,}  ({(time.time()-t0)/60:.1f} min)")
    return skeleton


# Stage 3: STRING PPI filter 

def load_ppi(gene_list: list):
    """
    Load STRING PPI pairs where both genes appear in the expression matrix.
    Returns (ppi_set, ppi_genes).
    Returns (None, all_genes) if PPI_FILE is None or not found — pipeline
    then falls back to two-stage intersection (MB-LASSO ∩ DNN).
    """
    if PPI_FILE is None or not os.path.exists(PPI_FILE):
        reason = "PPI_FILE set to None" if PPI_FILE is None else f"PPI file not found: {PPI_FILE}"
        print(f"\n  [{reason}]")
        print(f"  Continuing without PPI filter (two-stage pipeline).")
        return None, set(gene_list)

    print(f"\n[Stage 3] Loading STRING PPI (score >= {PPI_SCORE_THRESHOLD})...")
    df = pd.read_csv(PPI_FILE, sep="\t")

    cols      = df.columns.tolist()
    col_a     = next((c for c in cols if "gene" in c.lower() and "a" in c.lower()), cols[0])
    col_b     = next((c for c in cols if "gene" in c.lower() and "b" in c.lower()), cols[1])
    score_col = next((c for c in cols if "score" in c.lower()), None)

    if score_col:
        df = df[df[score_col] >= PPI_SCORE_THRESHOLD]

    gene_set  = set(gene_list)
    ppi_set   = set()
    ppi_genes = set()

    for _, row in df.iterrows():
        g1, g2 = str(row[col_a]).strip(), str(row[col_b]).strip()
        if g1 in gene_set and g2 in gene_set and g1 != g2:
            key = "___".join(sorted([g1, g2]))
            ppi_set.add(key)
            ppi_genes.add(g1)
            ppi_genes.add(g2)

    print(f"  PPI pairs in matrix    : {len(ppi_set):,}")
    print(f"  Genes with PPI support : {len(ppi_genes):,} / {len(gene_list):,}")
    print(f"  Genes skipped (no PPI) : {len(gene_list) - len(ppi_genes):,}")
    return ppi_set, ppi_genes


#  Stage 4: DNN knockoff selection 

def _build_skeleton_neighbours(skeleton: set) -> dict:
    """Build per-gene neighbour lookup from the skeleton edge set."""
    nbrs = {}
    for key in skeleton:
        a, b = key.split("___")
        nbrs.setdefault(a, set()).add(b)
        nbrs.setdefault(b, set()).add(a)
    return nbrs


def _collect_dnn_results(dnn_dir: str) -> set:
    """Read all per-gene DNN CSV files and return the union of selected edges."""
    assoc_set = set()
    for fname in os.listdir(dnn_dir):
        if not fname.startswith("DNN_") or not fname.endswith(".csv"):
            continue
        fpath = os.path.join(dnn_dir, fname)
        if os.path.getsize(fpath) == 0:
            continue
        try:
            df = pd.read_csv(fpath)
        except pd.errors.EmptyDataError:
            continue
        if df.empty:
            continue
        for _, row in df.iterrows():
            key = "___".join(sorted([str(row["Feature1"]), str(row["Feature2"])]))
            assoc_set.add(key)
    return assoc_set


def run_dnn(dataset: pd.DataFrame, ppi_genes: set, skeleton: set) -> set:
    """
    Run DNN knockoff selection on PPI-covered genes.

    Only skeleton neighbours are used as DNN inputs per gene (not all p genes).
    This is statistically valid (non-neighbours have near-zero knockoff statistics
    after MB-LASSO conditioning) and gives ~100-500x speedup.

    Results are checkpointed per gene — safe to interrupt and resume.
    """
    from DL.DNN.DNN import DNN
    from DL.FDR.FDR_control import FDR_control

    ko_path = os.path.join(DATA_FOLDER, KNOCKOFF_FILE)
    if not os.path.exists(ko_path):
        raise FileNotFoundError(
            f"Knockoff matrix not found: {ko_path}\n"
            f"Stage 1 must complete successfully before Stage 4."
        )

    print("\n[Stage 4] DNN knockoff selection...")

    ko_full     = pd.read_csv(ko_path, sep="\t", index_col=0)
    col_list    = dataset.columns.tolist()
    col_idx_map = {c: i for i, c in enumerate(col_list)}
    p_full      = len(col_list)
    n_samp      = dataset.shape[0]
    x_orig_full = ko_full.values[:, :p_full].astype(np.float32)
    x_ko_full   = ko_full.values[:, p_full:].astype(np.float32)
    del ko_full

    skeleton_nbrs = _build_skeleton_neighbours(skeleton)
    nbr_sizes     = [len(v) for v in skeleton_nbrs.values()]
    avg_nbrs      = int(np.mean(nbr_sizes)) if nbr_sizes else 0

    print(f"  n={n_samp}  |  p={p_full:,}")
    print(f"  PPI-covered genes       : {len(ppi_genes):,}")
    print(f"  Avg skeleton neighbours : {avg_nbrs} per gene")

    dnn_dir = os.path.join(DATA_FOLDER, "DNNSelection")
    os.makedirs(dnn_dir, exist_ok=True)

    target_genes  = [c for c in col_list if c in ppi_genes and c in skeleton_nbrs]
    existing_csvs = {
        f.replace("DNN_", "").replace(".csv", "")
        for f in os.listdir(dnn_dir)
        if f.startswith("DNN_") and f.endswith(".csv")
    }
    new_genes = [c for c in target_genes if c not in existing_csvs]

    print(f"  Target genes            : {len(target_genes):,}")
    print(f"  Already computed        : {len(existing_csvs):,}")
    print(f"  To run now              : {len(new_genes):,}")

    if not new_genes:
        print("  All genes done — collecting results.")
        assoc_set = _collect_dnn_results(dnn_dir)
        print(f"  DNN associations: {len(assoc_set):,}")
        return assoc_set

    completed = load_checkpoint()
    if existing_csvs and not completed:
        print(f"  Checkpoint missing — rebuilding from {len(existing_csvs):,} existing CSVs...")
        completed = existing_csvs
        save_checkpoint(completed)

    remaining = [c for c in new_genes if c not in completed]
    print(f"  Genes remaining         : {len(remaining):,}")

    t0 = time.time()

    for gene_idx, col_name in enumerate(tqdm(remaining, desc="  DNN", unit="gene")):
        nbrs     = sorted(skeleton_nbrs.get(col_name, set()) - {col_name})
        keep_idx = [col_idx_map[g] for g in nbrs if g in col_idx_map]
        nbrs     = [col_list[i] for i in keep_idx]
        p_num    = len(keep_idx)

        if p_num == 0:
            completed.add(col_name)
            save_checkpoint(completed)
            continue

        x_3d        = np.zeros((n_samp, p_num, 2), dtype=np.float32)
        x_3d[:,:,0] = x_orig_full[:, keep_idx]
        x_3d[:,:,1] = x_ko_full[:, keep_idx]
        y_vals      = dataset[[col_name]].values.astype(np.float32)
        coeff       = float(np.float32(0.05 * np.sqrt(2.0 * np.log(p_num) / n_samp)))

        result_dir  = os.path.join(TMP_DIR, f"DNN_{col_name}")
        x_data_path = os.path.join(TMP_DIR, f"X_{col_name}.txt")
        os.makedirs(result_dir, exist_ok=True)

        pd.DataFrame(x_orig_full[:, keep_idx], columns=nbrs).to_csv(
            x_data_path, sep="\t", index=False
        )

        model    = DNN().build_DNN(p_num, 1, coeff)
        callback = DNN.Job_finish_Callback(result_dir, p_num)
        DNN().train_DNN(model, x_3d, y_vals, callback,
                        epochs=DNN_EPOCHS, batch_size=DNN_BATCH_SIZE)

        # Clear TF session every 200 genes to prevent memory accumulation
        if (gene_idx + 1) % 200 == 0:
            tf.keras.backend.clear_session()

        selected = FDR_control().controlFilter(
            x_data_path, result_dir, offset=FDR_OFFSET, q=FDR_Q
        )

        assoc = [
            {"Feature1": (f[0] if isinstance(f, (list, tuple)) else f),
             "Feature2": col_name}
            for f in selected
        ]
        pd.DataFrame(assoc).to_csv(
            os.path.join(dnn_dir, f"DNN_{col_name}.csv"), index=False
        )

        shutil.rmtree(result_dir, ignore_errors=True)

        completed.add(col_name)
        save_checkpoint(completed)

        if (gene_idx + 1) % 100 == 0:
            elapsed  = time.time() - t0
            per_gene = elapsed / (gene_idx + 1)
            eta_h    = per_gene * (len(remaining) - gene_idx - 1) / 3600
            print(f"\n  [{gene_idx+1}/{len(remaining)}]  "
                  f"{per_gene:.1f}s/gene  |  ETA {eta_h:.1f}h")

    assoc_set = _collect_dnn_results(dnn_dir)
    print(f"\n  DNN associations: {len(assoc_set):,}")
    clear_checkpoint()
    return assoc_set


#  Stage 5: Intersection 

def intersect_all(skeleton: set, dnn_assoc: set, ppi_set) -> list:
    """
    Three-way intersection: MB-LASSO ∩ DNN ∩ STRING PPI.
    Falls back to two-stage intersection if ppi_set is None.
    """
    print(f"\n[Stage 5] Intersection...")
    print(f"  MB-LASSO skeleton : {len(skeleton):,}")
    print(f"  DNN associations  : {len(dnn_assoc):,}")

    two_stage = skeleton & dnn_assoc
    print(f"  LASSO ∩ DNN       : {len(two_stage):,}")

    if ppi_set is None:
        final = list(two_stage)
        print(f"  Final (no PPI)    : {len(final):,}")
    else:
        print(f"  STRING PPI pairs  : {len(ppi_set):,}")
        final = list(two_stage & ppi_set)
        print(f"  Final (all three) : {len(final):,}")

    if len(final) == 0:
        print("\n  WARNING: No edges survived intersection.")
        print("  Try: lower MB_ALPHA, lower PPI_SCORE_THRESHOLD, or lower FDR_Q")

    return final


# Stage 6: Inverse covariance 

def compute_inverse_covariance(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the precision matrix of the expression data.
    Used for edge orientation (DG Score) and regulation sign labelling.
    GPU-accelerated via CuPy if available.
    """
    print("\nComputing inverse covariance...")
    cov_np = dataset.cov().values.astype(np.float64)
    if CUPY_AVAILABLE:
        print("  Using CuPy (GPU)...")
        inv_np = cp.asnumpy(cp.linalg.pinv(cp.asarray(cov_np)))
    else:
        print(f"  Using NumPy ({N_CORES} threads)...")
        inv_np = np.linalg.pinv(cov_np)
    return pd.DataFrame(inv_np, index=dataset.columns, columns=dataset.columns)


#  Stage 7: Edge orientation 

def orient_edges(dataset: pd.DataFrame,
                 associations: list,
                 corr_inv: pd.DataFrame) -> nx.DiGraph:
    """
    Orient each undirected association using the Degenerate Gaussian Score.
    For each pair (A, B): compare localScore(A|B) vs localScore(B|A).
    The direction with the lower score (better fit) wins.
    Edge weight = score difference (used for cycle breaking).
    """
    from causal.DegenerateGaussianScore import DegenerateGaussianScore

    print(f"\n[Stage 6] Orienting {len(associations):,} edges...")
    col_map   = {col: i for i, col in enumerate(dataset.columns)}
    dg        = DegenerateGaussianScore(dataset, discrete_threshold=DISCRETE_THRESHOLD)
    graph     = nx.DiGraph()
    oriented  = tied = skipped = 0

    for key in tqdm(associations, desc="  Orienting", unit="edge"):
        f1, f2 = key.split("___")

        if f1 not in col_map or f2 not in col_map:
            skipped += 1; continue

        try:
            if abs(float(corr_inv.loc[f1, f2])) <= 0.0:
                skipped += 1; continue
        except Exception:
            skipped += 1; continue

        s1 = dg.localScore(col_map[f1], {col_map[f2]})
        s2 = dg.localScore(col_map[f2], {col_map[f1]})

        if s1 < s2:
            graph.add_edge(f2, f1, weight=s2 - s1); oriented += 1
        elif s1 > s2:
            graph.add_edge(f1, f2, weight=s1 - s2); oriented += 1
        else:
            tied += 1

    print(f"  Oriented: {oriented:,}  |  Tied: {tied:,}  |  Skipped: {skipped:,}")
    return graph


#  Stage 8: DAG enforcement 

def remove_cycles(graph: nx.DiGraph) -> nx.DiGraph:
    """
    Iteratively remove the weakest edge (lowest DG Score weight) from each cycle
    until the graph is acyclic.
    """
    print("\n[Stage 7] Enforcing DAG...")
    cycles  = list(nx.simple_cycles(graph))
    removed = 0
    while cycles:
        cycle = cycles[0]
        weak  = min(
            ((cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))),
            key=lambda e: graph[e[0]][e[1]]["weight"] if graph.has_edge(*e) else float("inf")
        )
        if graph.has_edge(*weak):
            graph.remove_edge(*weak)
            removed += 1
        cycles = list(nx.simple_cycles(graph))
    print(f"  Removed {removed:,} cycle edges. DAG complete.")
    return graph


#  Stage 9: Save 

def save_network(graph: nx.DiGraph, dataset: pd.DataFrame, corr_inv: pd.DataFrame):
    """
    Label edges with regulation direction (activation / repression) using
    the precision matrix (partial correlation), then save to CSV.
    Falls back to Pearson if a pair is missing from corr_inv.
    """
    print("\nLabelling and saving...")
    edge_list = []

    for cause, effect in tqdm(graph.edges(), desc="  Labelling", unit="edge"):
        w = graph[cause][effect]["weight"]
        try:
            pcor = -float(corr_inv.loc[cause, effect])
            reg  = ("Positive (Activation)" if pcor > 0
                    else "Negative (Repression)" if pcor < 0
                    else "Undefined")
        except Exception:
            corr = scipy.stats.pearsonr(dataset[cause].values, dataset[effect].values)[0]
            reg  = ("Positive (Activation)" if corr > 0
                    else "Negative (Repression)" if corr < 0
                    else "Undefined")

        edge_list.append({
            "Cause":      cause,
            "Effect":     effect,
            "EffectSize": float(np.log(w)) if w > 0 else float("nan"),
            "Regulation": reg,
        })

    out_path = os.path.join(DATA_FOLDER, OUTPUT_FILE)
    pd.DataFrame(edge_list).to_csv(out_path, index=False)
    print(f"\nNetwork saved : {out_path}")
    print(f"Total edges   : {len(edge_list):,}")
    print(f"Total genes   : {graph.number_of_nodes():,}")


#  Main 

if __name__ == "__main__":
    _validate_config()

    print(f"\nGlobal Causal Network Pipeline  (DAG-deepVASE adapted)")
    print(f"GPU : {'yes (' + str(len(gpus)) + ' device(s))' if gpus else 'no'}")
    print(f"CPU : {N_CORES} core(s)")

    dataset = load_dataset()
    n, p    = dataset.shape
    print(f"\nWorking shape: {n:,} samples × {p:,} genes")

    if n < 500:
        print(f"\n  Note: n={n} is small relative to p={p:,}.")
        print(f"  Using LedoitWolf knockoffs and MB_ALPHA={MB_ALPHA} to compensate.")

    generate_knockoffs(dataset)
    skeleton              = build_mb_skeleton(dataset)
    ppi_set, ppi_genes    = load_ppi(dataset.columns.tolist())
    dnn_assoc             = run_dnn(dataset, ppi_genes, skeleton)
    final_assoc           = intersect_all(skeleton, dnn_assoc, ppi_set)

    if not final_assoc:
        print("\nNo edges survived intersection. Exiting.")
        sys.exit(0)

    corr_inv = compute_inverse_covariance(dataset)
    graph    = orient_edges(dataset, final_assoc, corr_inv)
    graph    = remove_cycles(graph)
    save_network(graph, dataset, corr_inv)
