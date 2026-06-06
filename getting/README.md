# Bioinformatics AI Portfolio · Projects 1–3

This folder contains three standalone projects described in the root `PLANNING.md` and `TASK.md` files.

| Project | Folder | Description |
|---------|--------|-------------|
| **LLM-Gene** | `llm-gene/` | RAG Q&A over ClinVar + UniProt |
| **BioMultiAgent** | `biomultiagent/` | LangGraph multi-agent bioinformatics platform |
| **FedGenome** | `fedgenome/` | Federated variant pathogenicity classification |

**Project 0** (RNAStructFormer) lives at the repository root: `rna_structure_predictor.py`.

## Recommended run order

1. **RNAStructFormer** (root) — Kaggle GPU
2. **LLM-Gene** — CPU, download + index build
3. **BioMultiAgent** — CPU, needs Entrez email for PubMed
4. **FedGenome** — CPU/GPU, ClinVar download + FL simulation
