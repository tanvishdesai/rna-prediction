# RNAStructFormer · Transformer-based RNA 3D Backbone Predictor

> **Part of the Bioinformatics AI Portfolio** — Project 0 of 4.
> See [PLANNING.md](PLANNING.md) for the full portfolio roadmap.

Predicts 3D atomic coordinates for 5 backbone atoms per nucleotide
(P, C4′, C3′, O3′, C1′) from RNA sequences using a MSA-augmented Transformer encoder.

---

## Architecture

```
Input per residue
  Nucleotide token embedding   (d_model = 128)
  + MSA profile projection     (5 → d_model)
  → LayerNorm + sinusoidal positional encoding
  → 4× Pre-LN Transformer Encoder
       Multi-head self-attention (8 heads, GELU FFN, d_ff = 512)
  → LayerNorm → MLP head → (B, L, 15) coordinates
```

**Loss**: `L = L_coord(MSE on 15 coords) + 0.25 × L_dist(C3′ pairwise distance matrix)`

The distance term is rotation- and translation-invariant, addressing the key failure mode of raw coordinate regression.

**Key fixes over the BiLSTM baseline:**
- PAD masking bug fixed (PAD was index 4, `mask_zero` only guards index 0)
- MSA FASTA files are now actually parsed and fused into the representation
- Distance-matrix consistency loss for rotation-invariant training
- AdamW + cosine LR schedule + early stopping + gradient clipping

---

## Dataset — Stanford RNA 3D Folding (Kaggle)

This project uses the **[Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)** competition dataset.

**Files needed:**
| File | Description |
|------|-------------|
| `train_sequences.v2.csv` | Training RNA sequences |
| `train_labels.v2.csv` | 3D backbone atom coordinates |
| `validation_sequences.csv` | Validation sequences |
| `validation_labels.csv` | Validation coordinates |
| `test_sequences.csv` | Test sequences for submission |
| `MSA/*.MSA.fasta` | Multiple Sequence Alignments (evolutionary profiles) |

---

## Running on Kaggle (Recommended — Free GPU)

1. Go to [Kaggle](https://www.kaggle.com) → **Create Notebook**
2. **Add Data**: search for `Stanford RNA 3D Folding` (competition data)
3. **Enable GPU**: Settings → Accelerator → T4 GPU or P100
4. Upload `rna_structure_predictor.py` or paste the code
5. Install and run:

```python
!pip install torch pandas numpy tqdm -q
!python rna_structure_predictor.py
```

The script auto-detects `/kaggle/input/stanford-rna-3d-folding/` — no path changes needed.

**Where to update paths (if needed):**
- The `resolve_data_root()` function (line ~90) handles auto-detection
- Override with environment variable: `os.environ["RNA_DATA_DIR"] = "/your/path"`
- Or set directly in Config: `CFG.data_root = Path("/kaggle/input/stanford-rna-3d-folding")`

---

## Running Locally

```bash
# 1. Install dependencies
pip install torch pandas numpy tqdm

# 2. Set data directory (Windows)
set RNA_DATA_DIR=C:\path\to\stanford-rna-3d-folding

# 2. Set data directory (Linux/Mac)
export RNA_DATA_DIR=/path/to/stanford-rna-3d-folding

# 3. Run
python rna_structure_predictor.py
```

---

## Expected Output

```
18:05:00  INFO      Data root (Kaggle competition): /kaggle/input/stanford-rna-3d-folding
18:05:02  INFO      Parsing 3057 MSA FASTA files …
18:05:10  INFO        Cached 3057 MSA profiles.
18:05:12  INFO      RNAStructFormer: 1438607 trainable parameters (1.44 M)
18:05:13  INFO      ════ Training  ·  40 epochs  ·  device=cuda  ·  batch=16 ═══
18:05:45  INFO      Ep   1/40  train=2.3841  val=2.1203  RMSD=8.412 Å
18:06:18  INFO      Ep   5/40  train=1.7211  val=1.4882  RMSD=6.831 Å
...
18:35:00  INFO      Final validation RMSD: 5.XX Å
```

**Target**: Validation RMSD ≤ 6.0 Å

---

## Results

| Model | Val RMSD (Å) | Notes |
|-------|-------------|-------|
| BiLSTM baseline (broken PAD) | ~∞ | PAD masking bug |
| RNAStructFormer (this) | **TBD** | Run on Kaggle T4 GPU |

*Update this table after running.*

---

## Installation

```bash
pip install torch pandas numpy tqdm
```

PyTorch ≥ 2.0 recommended. GPU strongly recommended (CUDA or MPS).

---

## Repository Structure

```
rna_structure_predictor.py   Main training + inference script
PLANNING.md                  Full 4-project portfolio plan
TASK.md                      Active task board with todos
README.md                    This file
requirements.txt             Python dependencies
MSA/                         (not tracked) MSA FASTA files — download from Kaggle
```

---

## Upgrade Path

- Swap token embedding for frozen Nucleotide Transformer (`InstaDeepAI/nucleotide-transformer-v2-250m-multi-species`)
- Add base-pair interaction bias to attention weights
- Frame-aligned point error (FAPE) loss from AlphaFold2
- Federated extension: see `../fedgenome/` in the portfolio
