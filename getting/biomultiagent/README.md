# BioMultiAgent · Multi-Agent Bioinformatics Platform

LangGraph-orchestrated supervisor that routes natural-language queries to specialist agents:

| Agent | Capabilities |
|-------|-------------|
| **SeqAgent** | Translation, GC content, motif search |
| **AlignAgent** | Pairwise alignment, star MSA |
| **AnnotAgent** | ORF prediction (6 frames) |
| **PhyloAgent** | Neighbor-Joining tree + ASCII render |
| **LitAgent** | PubMed search via NCBI Entrez |

## Setup (run in this order)

```bash
cd getting/biomultiagent
pip install -r requirements.txt

# Set your email for NCBI Entrez (required for PubMed)
export ENTREZ_EMAIL=your@email.com        # Linux/Mac
$env:ENTREZ_EMAIL = "your@email.com"      # Windows PowerShell

# Step 1 — compound query demo (no LLM API needed)
python integration_demo.py

# Step 2 — Flask web UI
python app.py
# Open http://localhost:5001
```

## Kaggle notes

- Runs on CPU; no GPU required.
- Set `ENTREZ_EMAIL` in notebook secrets for LitAgent.
- Ollama is optional — routing uses keyword-based intent classification.

## Example compound query

> Translate this sequence, predict ORFs, align it to 3 homologs, build a phylogenetic tree, and find 5 recent PubMed papers.

## Architecture

```
User query → Supervisor (intent classify + decompose)
          → [Seq | Align | Annot | Phylo | Lit] agents
          → Result fusion + ChromaDB session memory
          → Final response + citations
```
