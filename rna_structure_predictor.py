"""
RNAStructFormer  ·  Transformer-based RNA 3D Backbone Structure Predictor
=========================================================================

Complete rewrite of the original BiLSTM baseline.

Changes
-------
  BUG FIX   PAD remapped to index 0 (original PAD='P'→4 silently bypassed
            mask_zero=True, so every training run learned on padding noise)
  FEATURE   MSA evolutionary profiles parsed from MSA/*.fasta and fused into
            the per-residue input representation
  FEATURE   Pre-LayerNorm Transformer encoder (4 layers, 8 heads, GELU FFN)
            replaces the Bidirectional LSTM
  FEATURE   Distance-matrix consistency loss — rotation-invariant pairwise
            C3′ distance term alongside coordinate MSE
  FEATURE   AdamW + cosine LR schedule + gradient clipping + early stopping
  METRIC    Per-residue RMSD (Å) reported on validation set every 5 epochs

Quick start (Kaggle — recommended)
----------------------------------
  1. Create a Kaggle notebook with GPU enabled.
  2. Add data source: Competition → "Stanford RNA 3D Folding"
     https://www.kaggle.com/competitions/stanford-rna-3d-folding
  3. Upload this script (or clone your GitHub repo) and run:
       pip install torch pandas numpy tqdm
       python rna_structure_predictor.py

  Data is auto-detected under /kaggle/input/stanford-rna-3d-folding/

Quick start (local)
-------------------
  Set RNA_DATA_DIR to your extracted competition folder, then run:
    set RNA_DATA_DIR=C:\\path\\to\\stanford-rna-3d-folding   # Windows
    export RNA_DATA_DIR=/path/to/stanford-rna-3d-folding    # Linux/Mac
    python rna_structure_predictor.py

Expected file layout (Stanford RNA 3D Folding competition)
----------------------------------------------------------
  train_sequences.v2.csv     train_labels.v2.csv   (labels: C1' x, y, z)
  validation_sequences.csv   validation_labels.csv
  test_sequences.csv
  MSA/   (*.fasta — evolutionary homolog alignments, one file per target)
  MSA_v2/  (expanded MSAs — merged with MSA/, v2 wins on ID clash)
  PDB_RNA/ (*.cif — PDB structures used for template-based modelling)

Competition format notes
------------------------
  Training labels store ONE C1' coordinate triplet per residue (x, y, z).
  Submission requires FIVE structure predictions per residue
  (x_1,y_1,z_1 … x_5,y_5,z_5) — best-of-5 TM-score evaluation.
"""

from __future__ import annotations

import logging
import math
import os
import re
import warnings
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("RNAStructFormer")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Configuration
# ═══════════════════════════════════════════════════════════════════════════════

def _auto_device() -> str:
    if torch.cuda.is_available():           return "cuda"
    if torch.backends.mps.is_available():   return "mps"
    return "cpu"


def resolve_data_root() -> Path:
    """
    Locate the Stanford RNA 3D Folding dataset.

    Search order:
      1. RNA_DATA_DIR environment variable (local override)
      2. /kaggle/input/stanford-rna-3d-folding  (Kaggle competition slug)
      3. Any sub-folder under /kaggle/input/ that contains train_sequences*.csv
      4. Current working directory (local fallback)
    """
    if env := os.environ.get("RNA_DATA_DIR"):
        root = Path(env)
        if root.exists():
            log.info("Data root (RNA_DATA_DIR): %s", root)
            return root
        log.warning("RNA_DATA_DIR='%s' not found — falling back to auto-detect.", env)

    kaggle_slug = Path("/kaggle/input/competitions/stanford-rna-3d-folding")
    if kaggle_slug.exists():
        log.info("Data root (Kaggle competition): %s", kaggle_slug)
        return kaggle_slug

    kaggle_base = Path("/kaggle/input")
    if kaggle_base.exists():
        for candidate in sorted(kaggle_base.iterdir()):
            if candidate.is_dir() and any(
                (candidate / name).exists()
                for name in ("train_sequences.v2.csv", "train_sequences.csv")
            ):
                log.info("Data root (Kaggle input scan): %s", candidate)
                return candidate

    log.info("Data root (local cwd): %s", Path(".").resolve())
    return Path(".")


def _out_dir() -> Path:
    """Writable output directory — /kaggle/working on Kaggle, else cwd."""
    kaggle_working = Path("/kaggle/working")
    return kaggle_working if kaggle_working.exists() else Path(".")


@dataclass
class Config:
    # ── paths (relative to data_root unless absolute) ─────────────────────────
    data_root    : Path  = field(default_factory=resolve_data_root)
    train_seq    : str   = "train_sequences.v2.csv"
    train_lab    : str   = "train_labels.v2.csv"
    val_seq      : str   = "validation_sequences.csv"
    val_lab      : str   = "validation_labels.csv"
    test_seq     : str   = "test_sequences.csv"
    msa_dirs     : Tuple[str, ...] = ("MSA", "MSA_v2")
    pdb_dir      : str   = "PDB_RNA"
    ckpt_path    : str   = field(default_factory=lambda: str(_out_dir() / "rnastruct_best.pt"))
    submission   : str   = field(default_factory=lambda: str(_out_dir() / "submission.csv"))

    def path(self, name: str) -> str:
        """Resolve a dataset-relative path against data_root."""
        p = Path(name)
        return str(p) if p.is_absolute() else str(self.data_root / p)

    # ── data ──────────────────────────────────────────────────────────────────
    max_seq_len  : int   = 512   # pad / truncate all sequences to this length
    n_coords     : int   = 3     # C1' atom (x, y, z) — competition label format
    n_submit     : int   = 5     # structure slots in submission.csv (best-of-5)
    tbm_min_sim  : float = 0.55  # min sequence identity for PDB template reuse

    # ── model ─────────────────────────────────────────────────────────────────
    d_model      : int   = 128   # transformer hidden dim; must be divisible by n_heads
    n_heads      : int   = 8     # attention heads  →  head dim = d_model / n_heads = 16
    n_layers     : int   = 4     # transformer encoder depth
    d_ff         : int   = 512   # feed-forward inner dimension
    dropout      : float = 0.10

    # ── training ──────────────────────────────────────────────────────────────
    batch_size   : int   = 16
    epochs       : int   = 40
    lr           : float = 3e-4
    weight_decay : float = 1e-2
    grad_clip    : float = 1.0
    dist_weight  : float = 0.25  # λ weight on distance-matrix loss term
    patience     : int   = 8     # early-stopping patience in epochs

    # ── misc ──────────────────────────────────────────────────────────────────
    seed         : int   = 42
    device       : str   = field(default_factory=_auto_device)


CFG = Config()


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Vocabulary
# ═══════════════════════════════════════════════════════════════════════════════

# PAD must be index 0 so that:
#   · nn.Embedding(padding_idx=0) zeros out the padding embedding
#   · key_padding_mask (True = ignore) aligns with (tokens == 0)
# Original code mapped PAD='P' to index 4 while mask_zero only guards index 0.
VOCAB:      Dict[str, int] = {"<PAD>": 0, "A": 1, "C": 2, "G": 3, "U": 4}
VOCAB_SIZE: int             = len(VOCAB)   # 5
PAD_IDX:    int             = 0
MSA_DIM:    int             = 5            # A / C / G / U / gap frequency columns


# ═══════════════════════════════════════════════════════════════════════════════
# §3  MSA evolutionary-profile library
# ═══════════════════════════════════════════════════════════════════════════════

class MSALibrary:
    """
    Parses all *.MSA.fasta files in *msa_dir* and serves per-file
    position-frequency matrices of shape (max_seq_len, 5).

    Each matrix column records the fraction of aligned sequences with
    A / C / G / U / gap at that alignment column — the core co-evolutionary
    signal used by state-of-the-art RNA and protein structure models.

    Falls back to a uniform-distribution profile (0.2 each) when no file
    can be matched to the requested target-id, so training proceeds even
    with incomplete MSA coverage.
    """

    _GAP_CHARS  = frozenset("-.")
    _NUC_TO_IDX : Dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}

    def __init__(self, msa_dirs: List[str], max_seq_len: int) -> None:
        self._L  = max_seq_len
        self._db : Dict[str, np.ndarray] = {}
        found = False
        for msa_dir in msa_dirs:
            if Path(msa_dir).exists():
                found = True
                self._build(Path(msa_dir))
        if not found:
            log.warning(
                "No MSA directories found (%s) – using uniform priors.",
                ", ".join(msa_dirs),
            )

    # ── private ───────────────────────────────────────────────────────────────

    def _build(self, d: Path) -> None:
        if not d.exists():
            log.warning(
                "MSA dir '%s' not found – using uniform priors for all sequences.", d
            )
            return
        files = sorted(d.glob("*.fasta"))
        log.info("Parsing %d MSA FASTA files …", len(files))
        for fp in files:
            key     = fp.stem.replace(".MSA", "").upper()
            profile = self._fasta_to_profile(fp)
            if profile is not None:
                self._db[key] = profile
        log.info("  Cached %d MSA profiles.", len(self._db))

    def _fasta_to_profile(self, fp: Path) -> Optional[np.ndarray]:
        seqs: List[str] = []
        buf:  List[str] = []
        try:
            with fp.open() as fh:
                for raw in fh:
                    line = raw.strip()
                    if not line:
                        continue
                    if line.startswith(">"):
                        if buf:
                            seqs.append("".join(buf).upper())
                            buf = []
                    else:
                        buf.append(line)
                if buf:
                    seqs.append("".join(buf).upper())
        except Exception as exc:
            log.debug("Could not read '%s': %s", fp.name, exc)
            return None

        if not seqs:
            return None

        aln_len = len(seqs[0])
        counts  = np.zeros((aln_len, 5), np.float32)   # A C G U gap

        for seq in seqs:
            if len(seq) != aln_len:
                continue                                 # skip malformed rows
            for pos, ch in enumerate(seq):
                if ch in self._GAP_CHARS:
                    counts[pos, 4] += 1.0
                elif ch in self._NUC_TO_IDX:
                    counts[pos, self._NUC_TO_IDX[ch]] += 1.0

        row_sums = counts.sum(1, keepdims=True).clip(min=1.0)
        profile  = counts / row_sums        # row-wise relative frequencies

        # Align length to max_seq_len
        L = profile.shape[0]
        if L >= self._L:
            return profile[: self._L]
        pad = np.full((self._L - L, 5), 0.2, np.float32)
        return np.vstack([profile, pad])

    # ── public ────────────────────────────────────────────────────────────────

    def get(self, target_id: str) -> np.ndarray:
        """Return (max_seq_len, 5) profile; uniform prior if not found."""
        key = target_id.upper()
        if key in self._db:
            return self._db[key]
        # Prefix / suffix fallback – handles ID scheme mismatches
        for k in self._db:
            if k.startswith(key) or key.startswith(k):
                return self._db[k]
        return np.full((self._L, 5), 0.2, np.float32)   # uninformative prior


# ═══════════════════════════════════════════════════════════════════════════════
# §3b  PDB template library  (lightweight TBM — top solutions rely on this)
# ═══════════════════════════════════════════════════════════════════════════════

_RNA_BASES = frozenset("ACGU")


class PDBTemplateLibrary:
    """
    Index C1' coordinates from PDB_RNA/*.cif for template-based fallback.

    Winners (e.g. 1st place team_cp) allocate submission slots to diverse
    TBM templates before filling remaining slots with deep-learning models.
    This lightweight indexer enables the same strategy without external tools.
    """

    _C1_NAMES = frozenset({"C1'", r"C1\'", "C1*"})

    def __init__(self, pdb_dir: str, max_seq_len: int) -> None:
        self._L   = max_seq_len
        self._db  : Dict[str, np.ndarray] = {}   # sequence → (L, 3)
        self._build(Path(pdb_dir))

    def _build(self, d: Path) -> None:
        if not d.exists():
            log.info("PDB dir '%s' not found – TBM templates disabled.", d)
            return
        files = sorted(d.glob("*.cif"))
        log.info("Indexing %d PDB mmCIF files for TBM …", len(files))
        for fp in files:
            for seq, coords in self._parse_cif(fp):
                if len(seq) < 8:
                    continue
                key = seq.upper()
                if key not in self._db or len(coords) > len(self._db[key]):
                    self._db[key] = coords
        log.info("  Cached %d PDB sequence templates.", len(self._db))

    def _parse_cif(self, fp: Path) -> List[Tuple[str, np.ndarray]]:
        try:
            text = fp.read_text(errors="replace")
        except Exception as exc:
            log.debug("Could not read '%s': %s", fp.name, exc)
            return []

        # Locate the _atom_site loop and its column headers.
        header_m = re.search(
            r"loop_\s*\n(_atom_site\.\S+(?:\n_atom_site\.\S+)*)",
            text,
            re.MULTILINE,
        )
        if not header_m:
            return []

        cols = [ln.strip() for ln in header_m.group(1).splitlines()]
        need = {"label_atom_id", "label_comp_id", "label_seq_id",
                "Cartn_x", "Cartn_y", "Cartn_z"}
        if not need.issubset(set(cols)):
            return []

        col_idx = {c: i for i, c in enumerate(cols)}
        data_start = header_m.end()
        lines = text[data_start:].splitlines()

        # Per-chain residue → C1' coordinate
        chains: Dict[str, Dict[int, Tuple[str, np.ndarray]]] = {}
        n_cols = len(cols)

        for raw in lines:
            if raw.startswith("#") or raw.startswith("_"):
                break
            if raw.startswith("loop_"):
                break
            parts = raw.split()
            if len(parts) < n_cols:
                continue
            atom = parts[col_idx["label_atom_id"]]
            if atom not in self._C1_NAMES:
                continue
            base = parts[col_idx["label_comp_id"]].upper()
            if len(base) != 1 or base not in _RNA_BASES:
                continue
            try:
                resid = int(parts[col_idx["label_seq_id"]])
                xyz   = np.array(
                    [parts[col_idx["Cartn_x"]],
                     parts[col_idx["Cartn_y"]],
                     parts[col_idx["Cartn_z"]]],
                    dtype=np.float32,
                )
            except (ValueError, IndexError):
                continue
            asym = (
                parts[col_idx["auth_asym_id"]]
                if "auth_asym_id" in col_idx else "A"
            )
            chains.setdefault(asym, {})[resid] = (base, xyz)

        out: List[Tuple[str, np.ndarray]] = []
        for residues in chains.values():
            if not residues:
                continue
            ordered = [residues[r][0] for r in sorted(residues)]
            coords  = np.stack(
                [residues[r][1] for r in sorted(residues)], axis=0
            )
            seq = "".join(ordered)
            L   = coords.shape[0]
            if L >= self._L:
                out.append((seq, coords[: self._L]))
            else:
                pad = np.zeros((self._L - L, 3), np.float32)
                out.append((seq, np.vstack([coords, pad])))
        return out

    def best_templates(
        self, query: str, n: int, min_sim: float
    ) -> List[np.ndarray]:
        """
        Return up to *n* diverse template coordinate arrays for *query*,
        ranked by sequence similarity (difflib ratio).
        """
        q = query.upper()[: self._L]
        if not self._db:
            return []

        scored: List[Tuple[float, str]] = []
        for seq, _ in self._db.items():
            ratio = SequenceMatcher(None, q, seq[: len(q)]).ratio()
            if ratio >= min_sim:
                scored.append((ratio, seq))
        scored.sort(reverse=True)

        results: List[np.ndarray] = []
        for _, seq in scored[: n * 3]:   # oversample, then deduplicate
            if len(results) >= n:
                break
            tmpl = self._db[seq][: len(q)]
            if len(tmpl) < len(q):
                pad  = np.zeros((len(q) - len(tmpl), 3), np.float32)
                tmpl = np.vstack([tmpl, pad])
            # Skip near-duplicate templates (same geometry)
            if results and np.allclose(tmpl, results[-1], atol=1.0):
                continue
            results.append(tmpl.astype(np.float32))
        return results[:n]


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def _encode_sequence(seq: str, max_len: int) -> np.ndarray:
    """Nucleotide string → int64 array, PAD=0, truncated / padded to max_len."""
    enc = np.array(
        [VOCAB.get(c, PAD_IDX) for c in seq.upper()[:max_len]],
        dtype=np.int64,
    )
    if len(enc) < max_len:
        enc = np.pad(enc, (0, max_len - len(enc)))   # zero-pads → PAD token ✓
    return enc


def _load_sequences(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("target_id", "sequence"):
        if col not in df.columns:
            raise ValueError(f"'{path}' is missing required column '{col}'.")
    return df


def _detect_coord_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect coordinate columns in a label CSV.

    Competition v2 labels:  x, y, z          (C1' training coordinates)
    Submission / legacy:     x_1…x_5, y_1…z_5 (5 structure slots × xyz)
    """
    slot_cols = [f"{ax}_{i}" for i in range(1, 6) for ax in ("x", "y", "z")]
    if all(c in df.columns for c in slot_cols):
        return slot_cols
    if all(c in df.columns for c in ("x", "y", "z")):
        return ["x", "y", "z"]
    raise ValueError(
        f"Unrecognised coordinate columns in label file. "
        f"Expected (x,y,z) or (x_1…z_5); got: {list(df.columns)}"
    )


def _load_labels(
    path: str, max_seq_len: int, n_coords: int
) -> Dict[str, np.ndarray]:
    """
    Parse label CSV.  ID column format:  "<target_id>_<resid>"

    Supports:
      · C1' training format (x, y, z)           → (max_seq_len, 3)
      · 5-slot submission format (x_1…z_5)      → uses slot 1 only for training

    Returns  {target_id: (max_seq_len, n_coords) float32}.
    """
    df    = pd.read_csv(path)
    split = df["ID"].str.rsplit("_", n=1, expand=True)
    df["_tid"]   = split[0]
    df["_resid"] = pd.to_numeric(split[1], errors="coerce")

    coord_cols = _detect_coord_columns(df)
    if coord_cols == ["x", "y", "z"]:
        if n_coords != 3:
            log.warning(
                "Label file '%s' has C1' (x,y,z) format; using 3 coords.",
                Path(path).name,
            )
        use_cols = ["x", "y", "z"]
    else:
        # Submission-style columns — train on structure slot 1 (C1' of prediction 1)
        use_cols = [f"{ax}_1" for ax in ("x", "y", "z")]
        if not all(c in coord_cols for c in use_cols):
            use_cols = coord_cols[:n_coords]
        log.info(
            "Label file '%s': using slot-1 coords %s for training.",
            Path(path).name, use_cols,
        )

    out: Dict[str, np.ndarray] = {}
    n_out = len(use_cols)
    for tid, grp in df.groupby("_tid"):
        grp  = grp.sort_values("_resid")
        raw  = grp[use_cols].values.astype(np.float32)   # (L, n_out)
        arr  = np.zeros((max_seq_len, n_out), np.float32)
        rows = min(len(raw), max_seq_len)
        arr[:rows] = raw[:rows]
        out[str(tid)] = arr
    return out


class RNADataset(Dataset):
    """
    Yields four tensors per sample:

      tokens      (max_seq_len,)           int64    nucleotide indices, PAD=0
      msa_profile (max_seq_len, MSA_DIM)   float32  evolutionary frequency matrix
      coords      (max_seq_len, n_coords)  float32  z-score normalised coordinates
      pad_mask    (max_seq_len,)           bool     True at every padding position

    Coordinate normalisation (z-score) is fit on the training set and
    propagated to val / test via coord_mean / coord_std arguments.
    """

    def __init__(
        self,
        seq_path:   str,
        lab_path:   Optional[str],
        msa_lib:    MSALibrary,
        cfg:        Config,
        coord_mean: Optional[np.ndarray] = None,
        coord_std:  Optional[np.ndarray] = None,
    ) -> None:
        self._msa = msa_lib
        self._cfg = cfg

        log.info("Loading %s …", Path(seq_path).name)
        seq_df = _load_sequences(seq_path)

        labels: Dict[str, np.ndarray] = {}
        if lab_path is not None:
            log.info("Loading %s …", Path(lab_path).name)
            labels = _load_labels(lab_path, cfg.max_seq_len, cfg.n_coords)

        self._records: List[Dict] = []
        skipped = 0
        for _, row in seq_df.iterrows():
            tid = str(row["target_id"])
            if lab_path is not None and tid not in labels:
                skipped += 1
                continue
            self._records.append({
                "tid": tid,
                "seq": str(row["sequence"]),
                "coords": labels.get(
                    tid,
                    np.zeros((cfg.max_seq_len, cfg.n_coords), np.float32),
                ),
            })

        if skipped:
            log.warning("  Skipped %d sequences with no matching labels.", skipped)
        log.info("  %d samples ready.", len(self._records))

        # Fit coordinate normalisation on this split; accept external for val/test
        if coord_mean is None:
            all_c           = np.stack([r["coords"] for r in self._records])
            self.coord_mean = all_c.mean((0, 1))                # (n_coords,)
            self.coord_std  = all_c.std((0, 1)).clip(min=1e-6)
        else:
            self.coord_mean = coord_mean
            self.coord_std  = (
                coord_std if coord_std is not None
                else np.ones(cfg.n_coords, np.float32)
            )

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        r       = self._records[idx]
        tokens  = _encode_sequence(r["seq"], self._cfg.max_seq_len)
        profile = self._msa.get(r["tid"])
        coords  = (r["coords"] - self.coord_mean) / self.coord_std
        mask    = tokens == PAD_IDX   # True = position to ignore

        return (
            torch.from_numpy(tokens),
            torch.from_numpy(profile),
            torch.from_numpy(coords.astype(np.float32)),
            torch.from_numpy(mask),
        )

    def target_ids(self) -> List[str]: return [r["tid"] for r in self._records]
    def sequences(self)  -> List[str]: return [r["seq"] for r in self._records]


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Model
# ═══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Fixed sinusoidal position encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pos   = torch.arange(max_len, dtype=torch.float).unsqueeze(1)    # (L, 1)
        omega = torch.exp(
            -math.log(10_000.0)
            * torch.arange(0, d_model, 2, dtype=torch.float)
            / d_model
        )                                                                  # (d/2,)
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * omega)
        pe[:, 1::2] = torch.cos(pos * omega[: d_model // 2])
        self.register_buffer("pe", pe.unsqueeze(0))                        # (1, L, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, : x.size(1)])


class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block.

    Pre-norm (LN before each sub-layer, not after) is more stable
    than post-norm across a wide range of learning rates and is the
    default in modern transformer implementations.

    Order: LN → MHA + residual → LN → FFN + residual.
    """

    def __init__(self, d: int, heads: int, d_ff: int, drop: float) -> None:
        super().__init__()
        self.ln1  = nn.LayerNorm(d)
        self.ln2  = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.ff   = nn.Sequential(
            nn.Linear(d, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, d),
            nn.Dropout(drop),
        )

    def forward(
        self,
        x:                torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed  = self.ln1(x)
        attn, _ = self.attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn
        return x + self.ff(self.ln2(x))


class RNAStructFormer(nn.Module):
    """
    MSA-augmented Transformer encoder for per-residue C1' coordinate regression.

    Input representation (per residue, fused before positional encoding):
      · Learned nucleotide embedding  (B, L) int64  → (B, L, d_model)
      · MSA profile linear projection (B, L, 5)     → (B, L, d_model)
      · Element-wise sum → LayerNorm → sinusoidal positional encoding

    Encoder:
      n_layers stacked TransformerBlocks with pre-LN and GELU feed-forward.

    Output head:
      LayerNorm → Linear(d, d) → GELU → Linear(d, 3)
      Produces (B, L, 3) C1' (x, y, z) predictions per residue.
    """

    def __init__(self, cfg: Config) -> None:
        super().__init__()
        d = cfg.d_model

        self.tok_emb  = nn.Embedding(VOCAB_SIZE, d, padding_idx=PAD_IDX)
        self.msa_proj = nn.Linear(MSA_DIM, d)
        self.in_norm  = nn.LayerNorm(d)
        self.pos_enc  = PositionalEncoding(d, cfg.max_seq_len, cfg.dropout)

        self.encoder = nn.ModuleList([
            TransformerBlock(d, cfg.n_heads, cfg.d_ff, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        self.out_norm = nn.LayerNorm(d)
        self.head     = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, cfg.n_coords),
        )

        self._init_weights()
        n = sum(p.numel() for p in self.parameters() if p.requires_grad)
        log.info("RNAStructFormer: %d trainable parameters (%.2f M)", n, n / 1e6)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(
        self,
        tokens:   torch.Tensor,    # (B, L) int64
        profile:  torch.Tensor,    # (B, L, 5) float32
        pad_mask: torch.Tensor,    # (B, L) bool  — True = ignore (padding)
    ) -> torch.Tensor:             # → (B, L, 3)
        x = self.tok_emb(tokens) + self.msa_proj(profile)
        x = self.pos_enc(self.in_norm(x))
        for block in self.encoder:
            x = block(x, key_padding_mask=pad_mask)
        return self.head(self.out_norm(x))


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def coord_loss(
    pred:   torch.Tensor,   # (B, L, 3)
    target: torch.Tensor,   # (B, L, 3)
    mask:   torch.Tensor,   # (B, L) bool — True = padding
) -> torch.Tensor:
    """Masked MSE over C1' (x, y, z), ignoring padding positions."""
    per_res = F.mse_loss(pred, target, reduction="none").mean(-1)   # (B, L)
    valid   = ~mask
    return (per_res * valid).sum() / valid.sum().clamp(min=1)


def distance_matrix_loss(
    pred:    torch.Tensor,   # (B, L, 3)
    target:  torch.Tensor,   # (B, L, 3)
    mask:    torch.Tensor,   # (B, L) bool
    max_pos: int = 128,      # subsample positions to bound memory
) -> torch.Tensor:
    """
    Inter-residue C1' distance matrix consistency loss.

    Raw coordinate MSE is not rotation-invariant.  Pairwise C1' distances
    are preserved under rigid-body motion, giving a rotation-invariant
    training signal aligned with the competition's TM-score metric.

    max_pos subsamples positions to keep the (B, S, S) distance matrix
    within comfortable memory bounds during training.
    """
    p_pos = pred               # (B, L, 3)  predicted C1'
    t_pos = target             # (B, L, 3)  true C1'

    valid = (~mask).float().unsqueeze(-1)   # (B, L, 1)
    p_c3  = p_pos * valid
    t_c3  = t_pos * valid

    # Subsample positions to avoid O(L²) memory blow-up
    L = p_c3.size(1)
    if L > max_pos:
        idx   = torch.randperm(L, device=p_c3.device)[:max_pos]
        p_c3  = p_c3[:, idx]
        t_c3  = t_c3[:, idx]
        valid = valid[:, idx]

    # Pairwise Euclidean distances  →  (B, S, S)
    p_dist = (p_c3.unsqueeze(2) - p_c3.unsqueeze(1)).norm(dim=-1)
    t_dist = (t_c3.unsqueeze(2) - t_c3.unsqueeze(1)).norm(dim=-1)

    v         = valid.squeeze(-1)                      # (B, S)
    pair_mask = v.unsqueeze(2) * v.unsqueeze(1)        # (B, S, S)
    n_pairs   = pair_mask.sum().clamp(min=1)
    return ((p_dist - t_dist).pow(2) * pair_mask).sum() / n_pairs


def total_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    mask:   torch.Tensor,
    lam:    float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lc = coord_loss(pred, target, mask)
    ld = distance_matrix_loss(pred, target, mask)
    return lc + lam * ld, lc, ld


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Evaluation metric  (per-residue RMSD in Ångströms)
# ═══════════════════════════════════════════════════════════════════════════════

def per_sample_rmsd(
    pred_dn: np.ndarray,   # (L, 3) denormalised C1'
    true_dn: np.ndarray,   # (L, 3) denormalised C1'
    valid:   np.ndarray,   # (L,) bool — True = non-padding
) -> float:
    p = pred_dn[valid]   # (n_valid, 3)
    t = true_dn[valid]
    return float(np.sqrt(((p - t) ** 2).sum(-1).mean()))


@torch.no_grad()
def evaluate_rmsd(
    model:  nn.Module,
    loader: DataLoader,
    mean:   np.ndarray,
    std:    np.ndarray,
    device: str,
) -> float:
    model.eval()
    scores: List[float] = []
    for tokens, profile, coords, mask in loader:
        pred = model(
            tokens.to(device), profile.to(device), mask.to(device)
        ).cpu().numpy()
        true = coords.numpy()
        vm   = ~mask.numpy()
        for b in range(pred.shape[0]):
            pd_ = pred[b] * std + mean
            td_ = true[b] * std + mean
            scores.append(per_sample_rmsd(pd_, td_, vm[b]))
    return float(np.mean(scores)) if scores else float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Training
# ═══════════════════════════════════════════════════════════════════════════════

def _iter(loader: DataLoader, desc: str):
    return tqdm(loader, desc=desc, leave=False) if _TQDM else loader


def _to(cfg: Config, *tensors: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    return tuple(t.to(cfg.device) for t in tensors)


def train_epoch(
    model:  nn.Module,
    loader: DataLoader,
    opt:    torch.optim.Optimizer,
    cfg:    Config,
) -> Tuple[float, float, float]:
    model.train()
    tot = c_tot = d_tot = n = 0.0
    for tokens, profile, coords, mask in _iter(loader, "  train"):
        tokens, profile, coords, mask = _to(cfg, tokens, profile, coords, mask)
        opt.zero_grad()
        pred       = model(tokens, profile, mask)
        lt, lc, ld = total_loss(pred, coords, mask, cfg.dist_weight)
        lt.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        b      = tokens.size(0)
        tot   += lt.item() * b
        c_tot += lc.item() * b
        d_tot += ld.item() * b
        n     += b
    return tot / n, c_tot / n, d_tot / n


@torch.no_grad()
def val_epoch(
    model:  nn.Module,
    loader: DataLoader,
    cfg:    Config,
) -> Tuple[float, float, float]:
    model.eval()
    tot = c_tot = d_tot = n = 0.0
    for tokens, profile, coords, mask in loader:
        tokens, profile, coords, mask = _to(cfg, tokens, profile, coords, mask)
        pred       = model(tokens, profile, mask)
        lt, lc, ld = total_loss(pred, coords, mask, cfg.dist_weight)
        b      = tokens.size(0)
        tot   += lt.item() * b
        c_tot += lc.item() * b
        d_tot += ld.item() * b
        n     += b
    return tot / n, c_tot / n, d_tot / n


def fit(
    model:     nn.Module,
    tr_loader: DataLoader,
    vl_loader: DataLoader,
    cfg:       Config,
    mean:      np.ndarray,
    std:       np.ndarray,
) -> None:
    opt   = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = CosineAnnealingLR(opt, T_max=cfg.epochs, eta_min=cfg.lr / 20)

    best_vl  = float("inf")
    patience = 0

    log.info("═" * 66)
    log.info("Training  ·  %d epochs  ·  device=%s  ·  batch=%d",
             cfg.epochs, cfg.device, cfg.batch_size)
    log.info("═" * 66)

    for ep in range(1, cfg.epochs + 1):
        tl, tc, td = train_epoch(model, tr_loader, opt, cfg)
        vl, vc, vd = val_epoch(model, vl_loader, cfg)
        sched.step()

        rmsd_str = ""
        if ep % 5 == 0 or ep == 1:
            vr       = evaluate_rmsd(model, vl_loader, mean, std, cfg.device)
            rmsd_str = f"  RMSD={vr:.3f} Å"

        log.info(
            "Ep %3d/%d  train=%.4f (c=%.4f d=%.4f)  val=%.4f (c=%.4f d=%.4f)%s",
            ep, cfg.epochs, tl, tc, td, vl, vc, vd, rmsd_str,
        )

        if vl < best_vl:
            best_vl = vl; patience = 0
            torch.save(
                {
                    "epoch"    : ep,
                    "state"    : model.state_dict(),
                    "val_loss" : best_vl,
                    "mean"     : mean,
                    "std"      : std,
                },
                cfg.ckpt_path,
            )
            log.info("  ✓ checkpoint saved  (val=%.4f)", best_vl)
        else:
            patience += 1
            if patience >= cfg.patience:
                log.info("Early stopping at epoch %d.", ep)
                break

    log.info("Training complete.  Best val loss: %.4f", best_vl)


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Inference & submission
# ═══════════════════════════════════════════════════════════════════════════════

def predict_and_save(
    model:    nn.Module,
    test_ds:  RNADataset,
    cfg:      Config,
    mean:     np.ndarray,
    std:      np.ndarray,
    pdb_lib:  Optional[PDBTemplateLibrary] = None,
) -> None:
    """
    Write submission.csv with cfg.n_submit structure slots per residue.

    Slot filling strategy (mirrors top competition pipelines):
      1. PDB templates (diverse TBM candidates) when available
      2. Model prediction for remaining slots
      3. Duplicate model prediction if fewer than n_submit sources exist
    """
    model.eval()
    loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for tokens, profile, _, mask in loader:
            out = model(
                tokens.to(cfg.device), profile.to(cfg.device), mask.to(cfg.device)
            )
            chunks.append(out.cpu().numpy())

    preds    = np.concatenate(chunks, 0)                          # (N, L, 3)
    preds_dn = preds * std[np.newaxis, np.newaxis] + mean[np.newaxis, np.newaxis]

    c_names  = [f"{ax}_{i}" for i in range(1, cfg.n_submit + 1) for ax in ("x", "y", "z")]
    rows: List[Dict] = []
    n_tbm_used = 0

    for i, (tid, seq) in enumerate(zip(test_ds.target_ids(), test_ds.sequences())):
        L = min(len(seq), cfg.max_seq_len)
        model_coords = preds_dn[i, :L]                            # (L, 3)

        # Collect up to n_submit diverse structure candidates
        slots: List[np.ndarray] = []
        if pdb_lib is not None:
            tbm = pdb_lib.best_templates(seq, cfg.n_submit, cfg.tbm_min_sim)
            if tbm:
                n_tbm_used += 1
            slots.extend(tbm)
        slots.append(model_coords)
        while len(slots) < cfg.n_submit:
            slots.append(model_coords.copy())
        slots = slots[: cfg.n_submit]

        for r in range(L):
            row = {"ID": f"{tid}_{r + 1}", "resname": seq[r], "resid": r + 1}
            flat = []
            for slot in slots:
                flat.extend(slot[r].tolist())
            row.update(zip(c_names, flat))
            rows.append(row)

    pd.DataFrame(rows, columns=["ID", "resname", "resid"] + c_names).to_csv(
        cfg.submission, index=False
    )
    log.info(
        "Submission written → %s  (%d rows, %d/%d targets used TBM templates)",
        cfg.submission, len(rows), n_tbm_used, len(test_ds),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve(*paths: str) -> str:
    """Return the first path that exists on disk."""
    for p in paths:
        if Path(p).exists():
            return p
    return paths[-1]    # last as fallback — will raise a clear error on open()


def main() -> None:
    torch.manual_seed(CFG.seed)
    np.random.seed(CFG.seed)

    # Resolve dataset paths under data_root; fall back to v1 filenames if v2 absent
    CFG.train_seq = _resolve(
        CFG.path(CFG.train_seq), CFG.path("train_sequences.csv")
    )
    CFG.train_lab = _resolve(
        CFG.path(CFG.train_lab), CFG.path("train_labels.csv")
    )
    CFG.val_seq   = CFG.path(CFG.val_seq)
    CFG.val_lab   = CFG.path(CFG.val_lab)
    CFG.test_seq  = CFG.path(CFG.test_seq)
    msa_dirs      = [CFG.path(d) for d in CFG.msa_dirs if Path(CFG.path(d)).exists()]
    if not msa_dirs:
        msa_dirs = [CFG.path(CFG.msa_dirs[0])]   # trigger uniform-prior warning
    pdb_dir       = CFG.path(CFG.pdb_dir)

    log.info("Data root : %s", CFG.data_root)
    log.info("Train     : %s + %s", CFG.train_seq, CFG.train_lab)
    log.info("Val       : %s + %s", CFG.val_seq,   CFG.val_lab)
    log.info("MSA dirs  : %s", msa_dirs)
    log.info("PDB dir   : %s", pdb_dir)
    log.info("Outputs   : %s, %s", CFG.ckpt_path, CFG.submission)
    log.info("Model     : d=%d  heads=%d  layers=%d  max_L=%d  coords=%d  device=%s",
             CFG.d_model, CFG.n_heads, CFG.n_layers, CFG.max_seq_len,
             CFG.n_coords, CFG.device)

    # MSA library (MSA + MSA_v2 merged)
    msa = MSALibrary(msa_dirs, CFG.max_seq_len)
    pdb = PDBTemplateLibrary(pdb_dir, CFG.max_seq_len)

    # Datasets
    tr_ds = RNADataset(CFG.train_seq, CFG.train_lab, msa, CFG)
    vl_ds = RNADataset(
        CFG.val_seq, CFG.val_lab, msa, CFG,
        coord_mean=tr_ds.coord_mean,
        coord_std =tr_ds.coord_std,
    )

    pin       = (CFG.device == "cuda")
    tr_loader = DataLoader(tr_ds, CFG.batch_size, shuffle=True,  num_workers=0, pin_memory=pin)
    vl_loader = DataLoader(vl_ds, CFG.batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    # Model
    model = RNAStructFormer(CFG).to(CFG.device)

    # Train
    fit(model, tr_loader, vl_loader, CFG, tr_ds.coord_mean, tr_ds.coord_std)

    # Reload best checkpoint and report final RMSD
    ckpt = torch.load(CFG.ckpt_path, map_location=CFG.device, weights_only=True)
    model.load_state_dict(ckpt["state"])
    final_rmsd = evaluate_rmsd(model, vl_loader, tr_ds.coord_mean, tr_ds.coord_std, CFG.device)
    log.info("━" * 66)
    log.info("Final validation RMSD: %.3f Å", final_rmsd)
    log.info("━" * 66)

    # Test inference → submission.csv
    if Path(CFG.test_seq).exists():
        ts_ds = RNADataset(
            CFG.test_seq, None, msa, CFG,
            coord_mean=tr_ds.coord_mean,
            coord_std =tr_ds.coord_std,
        )
        predict_and_save(
            model, ts_ds, CFG, tr_ds.coord_mean, tr_ds.coord_std, pdb_lib=pdb
        )
    else:
        log.warning("Test file '%s' not found — skipping inference.", CFG.test_seq)


if __name__ == "__main__":
    main()
