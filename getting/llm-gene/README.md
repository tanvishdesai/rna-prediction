# LLM-Gene · Bioinformatics RAG Q&A

Retrieval-Augmented Generation pipeline over **ClinVar** and **UniProt Swiss-Prot** for gene, variant, and protein questions.

## Architecture

```
NL query → PubMedBERT dense retrieval (top 20)
        → BM25 sparse retrieval (top 20)
        → Reciprocal Rank Fusion → cross-encoder rerank → top 5
        → Ollama Mistral generation with citations
```

## Setup (run in this order)

```bash
cd getting/llm-gene
pip install -r requirements.txt
ollama pull mistral          # optional; fallback extractive mode if missing

# Step 1 — download knowledge bases (~300 MB)
python scripts/download_data.py

# Step 2 — build FAISS index (~10–20 min CPU)
python build_index.py

# Step 3 — evaluate retrieval (optional)
python evaluate.py

# Step 4 — start Flask demo
python app.py
# Open http://localhost:5000
```

## Kaggle / Colab notes

- ClinVar and UniProt are downloaded from public FTP (no Kaggle dataset needed).
- Index building needs ~4 GB RAM; use a CPU runtime.
- Ollama is not available on Kaggle — the pipeline falls back to extractive answers.
- For GPU embedding speed-up, set `batch_size=64` in `build_index.py`.

## Example queries

- "What TP53 variants are pathogenic in breast cancer?"
- "BRCA1 pathogenic mutations"
- "CFTR variants and cystic fibrosis"

## Data sources

| Source | URL |
|--------|-----|
| ClinVar variant_summary | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz |
| UniProt Swiss-Prot | https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz |
