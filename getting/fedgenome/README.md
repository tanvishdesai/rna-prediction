# FedGenome · Federated Cancer Variant Classification

Ports precision-weighted federated aggregation (FedAlert-style) to multi-site genomic variant pathogenicity classification using **ClinVar** data with simulated non-IID hospital partitions.

## Dataset

| Source | Content | Access |
|--------|---------|--------|
| **ClinVar variant_summary** | Pathogenic / benign SNV labels | Public FTP (no registration) |
| **TCGA MAF** (upgrade path) | Multi-cancer somatic mutations | [GDC Portal](https://portal.gdc.cancer.gov) — free registration |

This demo uses ClinVar with 3 simulated sites partitioned by chromosome (non-IID by genomic region), matching the FedAlert non-IID spirit without requiring TCGA access.

## Architecture

```
Site 1 (chr 1-8)    Site 2 (chr 9-16)    Site 3 (chr 17-22,X,Y)
      │                    │                      │
 LocalGenomeNet         LocalGenomeNet         LocalGenomeNet
 (1D-CNN on DNA         (same)                 (same)
  context windows)
      └────────────────────┼──────────────────────┘
                           ▼
              FedGenomeStrategy (precision-weighted FedAvg)
                           ▼
                  Global variant classifier
```

## Setup (run in this order)

```bash
cd getting/fedgenome
pip install -r requirements.txt

# Step 1 — download ClinVar (~200 MB)
python scripts/download_data.py

# Step 2 — build non-IID site partitions
python prepare_data.py

# Step 3 — run federated experiment + ablation
python run_federation.py --strategy all --rounds 10
```

## Kaggle notes

- CPU runtime is sufficient (small CNN, ~16k variants).
- GPU optional for faster local training.
- Results saved to `results/ablation.json` and `results/ablation.png`.

## Evaluation metrics

| Method | Description |
|--------|-------------|
| **Centralized** | Train on all sites' data combined |
| **FedAvg** | Uniform gradient aggregation |
| **FedGenome** | Precision-weighted client aggregation |

Reported: global AUC-ROC, per-site AUC, equity gap (std of site AUCs).

## Upgrade to TCGA

1. Register at https://portal.gdc.cancer.gov
2. Download MAF files for TCGA-BRCA, TCGA-LUAD, TCGA-COAD
3. Replace `prepare_data.py` site assignment with cancer-type partitions
