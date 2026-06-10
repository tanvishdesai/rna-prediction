# Portfolio Run Guide

Execute these steps **in order** to build indexes, run experiments, and collect results for all four projects.

---

## Prerequisites

- Python 3.10+
- Internet access (ClinVar/UniProt/PubMed/Ensembl downloads)
- Optional: CUDA GPU (RNAStructFormer training, LLM-Gene index build)
- Optional: [MUSCLE](https://www.drive5.com/muscle/) on PATH (BioMultiAgent MSA)
- Optional: [Ollama](https://ollama.ai) with `mistral` (LLM-Gene generation, BioMultiAgent intent)

Set your NCBI email for PubMed (required by NCBI policy):

```powershell
$env:ENTREZ_EMAIL = "your.email@university.ac.in"
```

---

## Project 1 · LLM-Gene

```powershell
cd "llm-gene"
pip install -r requirements.txt

# 1. Download ClinVar + UniProt annotations (~500 MB, one-time)
python scripts/download_data.py

# 2. Build FAISS index (5–15 min CPU)
python build_index.py

# 3. Evaluate retrieval + answer grounding
python evaluate.py

# 4. Start web UI
python app.py
# Open http://127.0.0.1:5000
```

**Results to record:** MRR, retrieval grounding %, answer grounding % from `evaluate.py` output.

---

## Project 2 · BioMultiAgent

```powershell
cd "biomultiagent"
pip install -r requirements.txt

# Run unit tests
python -m pytest tests/ -v

# Integration demo (needs network for PubMed)
python integration_demo.py

# Web UI
python app.py
# Open http://127.0.0.1:5001
```

**Results to record:** Screenshot or terminal output showing translate + PubMed + compound query working.

---

## Project 3 · FedGenome

```powershell
cd "fedgenome"
pip install -r requirements.txt

# 1. Download ClinVar
python scripts/download_data.py

# 2. Build site partitions (fetches real Ensembl GRCh38 context — needs network, ~10–30 min)
python prepare_data.py

# 3. Run federated ablation (all strategies)
python run_federation.py --strategy all --rounds 10
```

**Results to record:** `results/ablation.json`, `results/ablation.png`, `results/convergence.png`

---

## Project 0 · RNAStructFormer

Requires Stanford RNA 3D Folding competition data (Kaggle recommended).

```powershell
cd ".."   # repo root
pip install torch pandas numpy tqdm

# Set data directory (local) or run on Kaggle GPU notebook
$env:RNA_DATA_DIR = "C:\path\to\stanford-rna-3d-folding"
python rna_structure_predictor.py
```

**Results to record:** Validation RMSD (Å), naive baseline RMSD, `submission.csv` (if test data present)

---

## Suggested outreach order

1. **LLM-Gene** — show eval metrics + live BRCA1/TP53 query
2. **BioMultiAgent** — show compound query demo
3. **RNAStructFormer** — show RMSD after Kaggle training run
4. **FedGenome** — show ablation table after `prepare_data.py` + `run_federation.py`

