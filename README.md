# RNAStructFormer · RNA 3D Backbone Structure Predictor

Transformer-based model that predicts 3D coordinates of five RNA backbone atoms per nucleotide (P, C4′, C3′, O3′, C1′), augmented with Multiple Sequence Alignment (MSA) evolutionary profiles.

## Dataset

**Kaggle: [Stanford RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding)**

| File | Description |
|------|-------------|
| `train_sequences.v2.csv` | Training RNA sequences |
| `train_labels.v2.csv` | Per-residue 3D coordinates (15 values = 5 atoms × xyz) |
| `validation_sequences.csv` | Validation sequences |
| `validation_labels.csv` | Validation coordinates |
| `test_sequences.csv` | Test sequences (inference only) |
| `MSA/*.fasta` | Evolutionary alignments per target |

Data is **not** stored in this repository. Add the competition as a Kaggle data source, or download via:

```bash
kaggle competitions download -c stanford-rna-3d-folding
```

## Architecture

```
Nucleotide embedding (d=128) + MSA profile projection (5→128)
  → LayerNorm + sinusoidal positional encoding
  → 4× Pre-LN Transformer encoder (8 heads, GELU FFN)
  → MLP head → (B, L, 15) coordinates

Loss = MSE(coords) + 0.25 × C3′ pairwise distance-matrix loss
```

## Run on Kaggle (recommended)

1. Create a new **Notebook** with **GPU** enabled (T4 or P100).
2. **Add data** → Competitions → **Stanford RNA 3D Folding**.
3. Upload `rna_structure_predictor.py` (or clone this repo).
4. Run:

```python
!pip install -q torch pandas numpy tqdm
!python rna_structure_predictor.py
```

The script auto-detects data at `/kaggle/input/stanford-rna-3d-folding/`.  
Outputs (`rnastruct_best.pt`, `submission.csv`) are written to `/kaggle/working/`.

### Path override (local)

Set `RNA_DATA_DIR` to your extracted competition folder:

```powershell
# Windows PowerShell
$env:RNA_DATA_DIR = "C:\path\to\stanford-rna-3d-folding"
python rna_structure_predictor.py
```

```bash
# Linux / Mac
export RNA_DATA_DIR=/path/to/stanford-rna-3d-folding
python rna_structure_predictor.py
```

## Run locally

```bash
pip install -r requirements.txt
export RNA_DATA_DIR=/path/to/extracted/competition/data   # required
python rna_structure_predictor.py
```

Training takes ~10–30 min on GPU for 40 epochs. Watch logs for `RMSD=X.XXX Å` every 5 epochs. Target validation RMSD: **≤ 6.0 Å**.

## Outputs

| File | Description |
|------|-------------|
| `rnastruct_best.pt` | Best checkpoint (model weights + coord normalisation stats) |
| `submission.csv` | Test-set coordinate predictions |

## Portfolio

This is **Project 0** in a bioinformatics AI portfolio. See `PLANNING.md` and `TASK.md` for Projects 1–3 (`getting/llm-gene`, `getting/biomultiagent`, `getting/fedgenome`).
