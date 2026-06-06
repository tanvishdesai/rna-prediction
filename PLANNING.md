# PLANNING.md · Bioinformatics AI Research Portfolio

> **Goal**: Build a credible, publication-ready bioinformatics AI portfolio
> to approach Surbhi Pawar (PhD Scholar, NMIMS, Aug 2025–present) for
> research collaboration. All projects bridge existing AI/ML expertise into
> computational biology, ordered by speed-to-demo and increasing complexity.
>
> **Contact trigger**: Reach out once two GitHub repos are live with results.
> Target: end of Week 6.

---

## Portfolio at a Glance

| # | Project | Core angle | Est. demo | Publishability |
|---|---------|------------|-----------|----------------|
| 0 | **RNAStructFormer** | Revived Kaggle project — Transformer + MSA | Week 1 | Low standalone / High as FL component |
| 1 | **LLM-Gene** | RAG pipeline over ClinVar + UniProt via BioBERT | Week 2 | Medium — Database or ISMB App Note |
| 2 | **BioMultiAgent** | Multi-agent extension of Surbhi's BioNLP Platform | Week 5 | High — Bioinformatics journal or BioNLP @ ACL |
| 3 | **FedGenome** | FedAlert FL framework ported to cancer genomic variants | Week 8 | High — Briefings in Bioinformatics / Nature MI |

---

## Project 0 · RNAStructFormer (Revived)

### Problem statement
Predict the 3D atomic coordinates of 5 backbone atoms per nucleotide
(P, C4′, C3′, O3′, C1′) for arbitrary RNA sequences, guided by evolutionary
co-variation signals from Multiple Sequence Alignments.

### Why this project
- Existing codebase in your repo, largely done — just needed fixing and
  upgrading (BiLSTM → Transformer, padding bug, unused MSA files).
- The MSA folder is already downloaded (hours of preprocessing preserved).
- Gives you a concrete RMSD number to put in the GitHub README before
  approaching Surbhi.

### Architecture
```
Input per residue
  Nucleotide token embedding    (d_model = 128)
  + MSA profile projection      (5 → d_model)
  → LayerNorm + sinusoidal positional encoding
  → 4× Pre-LN Transformer Encoder
       Multi-head self-attention (8 heads, GELU FFN, d_ff = 512)
  → LayerNorm → MLP head → (B, L, 15)  coordinates
```

**Loss**: `L = L_coord(MSE on all 15 coords) + 0.25 × L_dist(C3′ pairwise distance matrix)`
The distance term is rotation- and translation-invariant, addressing the key
failure mode of raw coordinate regression.

### Dataset
- **Kaggle: Stanford RNA 3D Folding** competition
  - `train_sequences.v2.csv`, `train_labels.v2.csv` — already downloaded
  - `validation_sequences.csv`, `validation_labels.csv`
  - `MSA/*.MSA.fasta` — PDB-sourced evolutionary alignments — **already downloaded**
  - Source: https://www.kaggle.com/competitions/stanford-rna-3d-folding

### Tech stack
| Component | Tool |
|-----------|------|
| Framework | PyTorch ≥ 2.0 |
| Data | pandas, numpy |
| Progress | tqdm |
| Platform | Kaggle notebook (free T4 GPU) |

### Success metric
- Validation RMSD ≤ 6.0 Å (competitive for this model class and dataset size)
- Submission CSV generated for test set

### Upgrade path (post-collaboration)
- Swap token embedding for frozen Nucleotide Transformer encoder
  (`InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` on HuggingFace)
- Add base-pair interaction bias to attention weights
- Extend to FedGenome: federated training across PDB RNA families

---

## Project 1 · LLM-Gene

### Problem statement
Researchers querying bioinformatics databases (ClinVar, UniProt, OMIM, PubMed)
must manually switch between fragmented query interfaces with different syntax.
Natural language questions cannot be answered directly from these structured
knowledge bases without significant expert curation.

### Proposed solution
A **Retrieval-Augmented Generation (RAG) pipeline** that accepts natural
language questions about genes, variants, or proteins and returns grounded,
citation-backed answers by querying live bioinformatics databases through
a unified retrieval layer.

This directly extends the RAG patterns from the existing FAQ chatbot work
(Gemini embeddings, HyDE, hybrid BM25+dense, cross-encoder reranking) into
the bioinformatics domain — only the knowledge source changes.

### Architecture
```
NL query ("What TP53 variants are pathogenic in breast cancer?")
  │
  ├─ BioBERT embedding → FAISS dense retrieval (top 20)
  ├─ BM25 sparse retrieval on gene symbols + variant IDs (top 20)
  │
  └─ Reciprocal Rank Fusion → top 40
       │
       └─ Cross-encoder reranker → top 5 passages
            │
            └─ Prompt assembly + Ollama (Mistral) / Claude API
                 │
                 └─ Answer + cited ClinVar/UniProt/PubMed sources
```

Optional sub-module (adds novelty):
```
Input: raw DNA/protein FASTA sequence
  → ESM-2 or Nucleotide Transformer embedding
  → Fine-tuned classification head
  → Pathogenicity label (pathogenic / benign / VUS) + confidence
```

### Datasets and knowledge bases
| Source | Content | Access |
|--------|---------|--------|
| **ClinVar FTP** | Variant–phenotype associations | Public, no registration |
| **UniProt Swiss-Prot** | Curated protein annotations | Public REST API |
| **PubMed** via Entrez | Biomedical abstracts | Free, 3 req/s unauthenticated |
| **OMIM API** | Gene–disease relationships | Free academic registration |
| **dbSNP FTP** | SNP annotations and allele frequencies | Public |

Download ClinVar:
```bash
wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
gunzip variant_summary.txt.gz
```

Download UniProt Swiss-Prot:
```bash
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/\
knowledgebase/complete/uniprot_sprot.fasta.gz
```

### Pre-trained models
| Model | Purpose | HuggingFace ID |
|-------|---------|----------------|
| **PubMedBERT** | Biomedical query + passage embedding | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract` |
| **BioBERT** | Alternative biomedical embeddings | `dmis-lab/biobert-v1.1` |
| **MS-MARCO MiniLM** | Cross-encoder reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Mistral (Ollama)** | Local answer generation | `ollama pull mistral` |
| **ESM-2** (optional) | Protein sequence classification | `facebook/esm2_t6_8M_UR50D` |

### Tech stack
```
pip install sentence-transformers faiss-cpu rank-bm25 biopython flask
ollama pull mistral
```

### Evaluation
- **MRR** (mean reciprocal rank) on 50 gold-standard Q&A pairs
- **Answer grounding rate**: % of claims linked to a retrieved passage
- Compile evaluation set from ClinVar documentation + UniProt help pages

### Collaboration angle with Surbhi
Her BioNLP Platform routes natural language queries to local BioPython
workflows. LLM-Gene adds a **retrieval layer over external databases** —
the exact gap she has ("limited external database connectivity" noted in her
report's Limitations section). A Flask endpoint compatible with her existing
UI makes integration frictionless.

### Publishability
- **Database journal** (Oxford, IF 4.5) — explicitly targets bioinformatics tools
- **ISMB Application Notes** — short papers, practical systems
- **BioNLP workshop at ACL** — if framed as NLP for computational biology

### Timeline
1–2 weeks to working prototype · 3–5 weeks to paper-ready

---

## Project 2 · BioMultiAgent

### Problem statement
Surbhi's BioNLP Platform uses a **single-LLM keyword router** that pattern-
matches user queries to one bioinformatics workflow at a time. It cannot:
- Execute compound tasks ("align AND annotate AND find papers")
- Handle ambiguous queries requiring clarification
- Maintain context across a session
- Access external databases (PubMed, UniProt) dynamically

Her own report's Future Scope lists "multi-agent AI systems" as an explicit
next step she has not yet implemented.

### Proposed solution
Replace the single-router architecture with a **LangGraph-orchestrated
multi-agent layer** where a Supervisor delegates to specialist agents, each
owning one workflow domain. Your DST-AgenticNet paper (multi-agent fusion for
healthcare AI, AICCON 2026) provides the direct methodological precedent.

### Architecture
```
User natural language query
  │
  ▼
Supervisor Agent  [Ollama Mistral / Claude Sonnet]
  ├─ Intent classification (JSON-structured output)
  ├─ Task decomposition into sub-tasks
  ├─ Agent routing
  └─ Result fusion + natural language synthesis
       │
       ├──▶ SeqAgent     BioPython — translation, GC content, motif search
       ├──▶ AlignAgent   Pairwise + MSA (Biopython pairwise2, MUSCLE CLI)
       ├──▶ AnnotAgent   ORF finder, codon tables, gene annotation
       ├──▶ PhyloAgent   NJ tree construction, ETE3 ASCII + Newick output
       └──▶ LitAgent     PubMed Entrez search + abstract summarisation
                │
                ▼
       Session Memory  [ChromaDB]
       (query history, result cache, biological context persistence)
```

LangGraph state schema (implement first — everything derives from this):
```python
from typing import TypedDict, List, Dict, Any

class BioAgentState(TypedDict):
    query         : str
    intent        : str          # seq | align | annot | phylo | literature | compound
    sub_tasks     : List[str]
    agent_results : Dict[str, Any]
    memory_context: List[Dict]
    final_response: str
    citations     : List[str]
```

### Resources — frameworks
| Tool | Purpose | Install |
|------|---------|---------|
| **LangGraph** | Agent orchestration with typed state machines | `pip install langgraph` |
| **LangChain Community** | Tool wrappers, Entrez, BLAST | `pip install langchain-community` |
| **Ollama** | Local LLM inference (already in BioNLP stack) | `ollama pull mistral` |
| **LiteLLM** | Abstraction layer (swap Ollama ↔ Claude API) | `pip install litellm` |
| **ChromaDB** | Vector session memory | `pip install chromadb` |

### Resources — bioinformatics
| Tool | Purpose | Install |
|------|---------|---------|
| **BioPython** | Core sequence analysis (already in BioNLP) | `pip install biopython` |
| **ETE3** | Phylogenetic tree rendering | `pip install ete3` |
| **MUSCLE** | Multiple sequence alignment (CLI, called as subprocess) | Conda or binary |
| **NCBI Entrez** (Biopython) | PubMed + GenBank retrieval | Included in BioPython |
| **BLAST+** | Sequence similarity search | NCBI local install or API |

### Key deliverables
```
bioagent/
  supervisor.py          LangGraph state machine, intent classifier
  agents/
    seq_agent.py         SeqAgent — BioPython sequence analysis
    align_agent.py       AlignAgent — pairwise + MSA
    annot_agent.py       AnnotAgent — ORF prediction, annotation
    phylo_agent.py       PhyloAgent — tree construction + rendering
    lit_agent.py         LitAgent — PubMed retrieval + summarisation
  memory/
    chroma_store.py      ChromaDB session memory layer
  tests/
    test_agents.py       Unit tests for each agent in isolation
  integration_demo.py    End-to-end: compound query demo script
```

### Collaboration strategy
Fork `Suru1496/Bio_NLP` → add `bioagent/` module → write a PR description
that explains the architectural upgrade and directly references her future
scope section. This converts a cold LinkedIn message into a concrete open-source
contribution she can evaluate independently.

Target demo query for the PR:
> "Translate this sequence, predict ORFs, run multiple sequence alignment
> against these 3 homologs, build a phylogenetic tree, and find 5 recent
> PubMed papers on this gene family."

One command. One response. Everything she envisioned as future work, running.

### Publishability
**High.** Frame the paper around:
- The multi-agent orchestration architecture for bioinformatics
- Compound task decomposition and result fusion
- Memory-augmented session continuity
- Comparison to single-LLM routing (BioNLP baseline)

Targets:
- **Bioinformatics** (Oxford, IF 5.8) — software/methods track
- **BioNLP workshop at ACL 2027** — NLP for computational biology
- **ISMB 2027** — systems and software track

Leverage DST-AgenticNet as direct prior for the multi-agent methodology.

### Estimated timeline
3–4 weeks to working compound-query demo · 6–8 weeks to paper-ready

---

## Project 3 · FedGenome

### Problem statement
Cancer genomic datasets (somatic mutation calls, gene expression profiles,
CNV segments) are distributed across hospitals and research institutions that
cannot share raw patient data due to HIPAA/GDPR constraints. Existing
centralised cancer genomics ML models (e.g. trained on TCGA) suffer from
distribution shift when deployed at new institutions with different patient
demographics, cancer subtypes, or sequencing platforms — the classic non-IID
problem in federated learning.

### Proposed solution
Port the **FedAlert precision-weighted consensus aggregation framework** —
already proven on non-IID histopathology image data — to multi-site cancer
genomic **somatic variant classification**. The mapping is direct:

| FedAlert | FedGenome |
|----------|-----------|
| Pathology image tiles | Mutation context sequences |
| Histopathology subtypes as non-IID source | Cancer types (BRCA/LUAD/COAD) as non-IID source |
| Tissue classifier | Variant pathogenicity classifier |
| Precision-weighted consensus aggregation | Same, applied to genomic gradients |
| Non-IID Dirichlet partition | Same (α = 0.5 across cancer subtypes) |

Same algorithm. Different biological modality. The novelty is the domain transfer
and the genomic-specific encoding pipeline.

### Architecture
```
Simulated Site 1             Simulated Site 2             Simulated Site 3
TCGA-BRCA partition          TCGA-LUAD partition          TCGA-COAD partition
       │                            │                            │
 LocalGenomeNet               LocalGenomeNet               LocalGenomeNet
 (CNN-1D or                   (same architecture)          (same architecture)
  Transformer on
  k-mer features)
       │ ΔW₁                        │ ΔW₂                        │ ΔW₃
       └────────────────────────────┼────────────────────────────┘
                                    ▼
                    FedAlert Consensus Aggregator
                    (precision-weighted FedAvg
                     + non-IID divergence correction)
                    (ported to Flower Strategy API)
                                    │
                             Global Genome Model
                        (multi-cancer variant classifier)
```

**LocalGenomeNet** — two implementations, ablate both:

Option A — CNN-1D (fast baseline):
```
k-mer frequency feature vector (k=3, 64 features)
→ Conv1D(64) → ReLU → MaxPool
→ Conv1D(128) → ReLU → GlobalAvgPool
→ Linear(128 → 2)  [pathogenic / benign]
```

Option B — Transformer (stronger, transfer learning):
```
Nucleotide Transformer frozen encoder
(InstaDeepAI/nucleotide-transformer-v2-250m-multi-species)
→ Linear classification head
→ Fine-tuned on local site data
```

### Datasets
| Dataset | Content | Access |
|---------|---------|--------|
| **TCGA via GDC** | Multi-cancer somatic MAF files | Free registration at portal.gdc.cancer.gov |
| **ClinVar** | Variant pathogenicity labels | Public FTP, no registration |
| **gnomAD** | Population variant frequencies (for negative class) | Public |

TCGA partitioning strategy (no IRB needed — this is public data):
```python
# Simulate federation by cancer type — same non-IID spirit as FedAlert
sites = {
    "site_1": tcga_brca_samples,    # Breast cancer
    "site_2": tcga_luad_samples,    # Lung adenocarcinoma
    "site_3": tcga_coad_samples,    # Colon adenocarcinoma
}
# Within each site, apply Dirichlet(α=0.5) on mutation burden quartiles
# for intra-site non-IID variation
```

Variant encoding pipeline:
```python
# For each MAF row: extract ±10 bp flanking genomic context
# → tokenize as nucleotide sequence → Nucleotide Transformer embedding
# Label: pathogenic (ClinVar P/LP) vs benign (gnomAD common variants, AF > 0.01)
```

### Pre-trained models
| Model | Purpose | HuggingFace ID |
|-------|---------|----------------|
| **Nucleotide Transformer v2** | DNA sequence encoding | `InstaDeepAI/nucleotide-transformer-v2-250m-multi-species` |
| **DNABERT-2** | Alternative DNA LM | `zhihan1996/DNABERT-2-117M` |

### Frameworks
| Tool | Purpose | Install |
|------|---------|---------|
| **Flower (flwr)** | FL framework with custom Strategy API | `pip install flwr[simulation]` |
| **PySyft** (optional) | Stronger privacy guarantees, DP support | `pip install syft` |
| PyTorch | Local model training | Standard |
| scikit-learn | AUC-ROC, F1, PR curves | Standard |

Flower aggregation integration point:
```python
import flwr as fl

class FedGenomeStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        # Port FedAlert's precision-weighted consensus here
        # Weight each client's gradients by its local precision score
        # Apply divergence penalty for highly non-IID updates
        ...
```

### Evaluation metrics
- Per-site AUC-ROC + global weighted AUC
- F1 macro (handles class imbalance in variant calls)
- **Equity gap**: std dev of site-wise AUC (lower = more equitable across sites)
- Communication round convergence curves
- Ablation table: FedAvg vs FedProx vs FedGenome (your method)

### Collaboration angle with Surbhi
Her GitHub has `HER2_Breast_Cancer_Target_Protein_Data` and
`BreastCancerPrediction` — she explicitly works on cancer genomics.
FedGenome is the privacy-preserving, multi-site version of exactly
what she's already doing. The pitch:
> "I extended my FedAlert FL framework to multi-site cancer genomic variant
> classification on TCGA. The non-IID handling maps directly from histopathology
> subtypes to cancer types. I think there's a natural paper combining this with
> your cancer genomics repos."

### Publishability
**High.**
- **Briefings in Bioinformatics** (Oxford, IF 9.5) — top bioinformatics journal
- **Nature Machine Intelligence** (high bar, high reward)
- **RECOMB 2027** — premier computational genomics conference
- **ICLR 2027 workshop**: FL + healthcare track

### Estimated timeline
4–6 weeks to working federated experiment · 8–12 weeks to paper-ready

---

## Tech Stack — Master Reference

| Layer | Tool | Notes |
|-------|------|-------|
| Deep learning | PyTorch ≥ 2.0 | All projects |
| Transformers | HuggingFace `transformers` | Pre-trained encoders |
| Bioinformatics | BioPython, ETE3 | Sequence + tree analysis |
| FL framework | Flower (flwr) | FedGenome |
| Agent framework | LangGraph | BioMultiAgent |
| Vector store | FAISS + ChromaDB | LLM-Gene + BioMultiAgent memory |
| LLM backend | Ollama (local) / Claude API | All agentic projects |
| Embeddings | BioBERT, PubMedBERT, ESM-2, Nucleotide Transformer | Project-specific |
| Web layer | Flask | Compatible with Surbhi's BioNLP stack |
| Experiment tracking | Weights & Biases (`wandb`) | Pick one and use it consistently |
| Version control | Git + GitHub (all repos public) | Required before outreach |

---

## Constraints

- **No institutional compute**: All projects must run on Kaggle (free T4/P100 GPU)
  or Google Colab. FedGenome can simulate federation on a single GPU.
- **No paid API dependency in demos**: Use Ollama for local LLM inference.
  Claude API is acceptable as an optional upgrade path.
- **All repos public**: Surbhi needs to be able to inspect code independently.
- **Time budget**: 8 weeks to contact trigger. Projects 0 and 1 are the minimum
  bar; Projects 2 and 3 strengthen the case after initial contact.

---

## Collaboration Timeline

```
Week 1    RNAStructFormer: run, record RMSD, push to GitHub with clean README
Week 1-2  LLM-Gene: ClinVar + UniProt index built, Flask demo live
Week 3-5  BioMultiAgent: supervisor + 5 agents + compound query demo
Week 4-8  FedGenome: TCGA partition + Flower FL + 3-site ablation results
Week 6    → Send LinkedIn message to Surbhi with 2 repos linked + 1 in progress
Week 8-10 → Target: first collaboration call or meeting
```
