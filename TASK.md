# TASK.md · Active Task Board

> **Last updated**: June 2026
> **Objective**: Bioinformatics AI portfolio for collaboration with Surbhi Pawar (NMIMS)
> **Contact trigger**: Two GitHub repos live with results → send LinkedIn message

---

## ✅ Completed

- [x] Diagnosed RNA prediction codebase — identified 3 critical issues
  - PAD=4 silently bypassing `mask_zero=True` (masking bug)
  - MSA FASTA files downloaded but never used
  - BiLSTM architecture unable to model long-range base-pair interactions
- [x] Full rewrite of `rna_structure_predictor.py`
  - Fixed PAD → index 0; key_padding_mask now works correctly
  - Replaced BiLSTM with pre-LN Transformer encoder (4 layers, 8 heads)
  - Added `MSALibrary` — parses all *.MSA.fasta files into frequency profiles
  - Added `distance_matrix_loss` — rotation-invariant C3′ pairwise distance term
  - Added `evaluate_rmsd` — reports Ångström RMSD on validation set every 5 epochs
  - AdamW + cosine LR + gradient clipping + early stopping + checkpointing
- [x] Created PLANNING.md — full architecture, datasets, tech stack for all 4 projects
- [x] Created TASK.md (this file)

---

## 🔴 Active — Week 1

---

### Project 0 · RNAStructFormer

**Goal**: Get a running model with a reportable RMSD number on the validation set.

#### Running the model

- [ ] **RNA-01** · Install dependencies on Kaggle / local
  ```bash
  pip install torch pandas numpy tqdm
  # Kaggle: enable GPU accelerator in notebook settings
  ```

- [ ] **RNA-02** · Run the updated script end-to-end
  ```bash
  python rna_structure_predictor.py
  ```
  - Watch for `RMSD=X.XXX Å` in logs at epoch 1, 5, 10, ...
  - Expect training to take 10–30 min on GPU for 40 epochs
  - A `rnastruct_best.pt` checkpoint file should appear after the first
    epoch where validation loss improves

- [ ] **RNA-03** · Record your baseline RMSD
  - The final log line prints: `Final validation RMSD: X.XXX Å`
  - Target: ≤ 6.0 Å (competitive for Transformer class on this dataset)
  - If RMSD > 10 Å after 10 epochs → check data paths and coordinate normalisation

- [ ] **RNA-04** · Sanity check — compute naive baseline RMSD
  - Predict `coord_mean` for every residue in every sequence
  - If model RMSD ≥ naive baseline RMSD → something is still wrong

- [ ] **RNA-05** · Tune if compute allows (Kaggle P100 recommended)
  ```python
  # In rna_structure_predictor.py, modify Config:
  CFG.d_model  = 256   # was 128
  CFG.n_layers = 6     # was 4
  # Expect ~3.5M params, ~5-10% RMSD improvement
  ```

#### GitHub cleanup

- [ ] **RNA-06** · Update `README.md`
  - Architecture section with ASCII diagram of the model pipeline
  - Installation: `pip install torch pandas numpy tqdm`
  - Run command and expected log output (paste first few epochs)
  - Results table: `| Model | Val RMSD (Å) |`
  - Note that MSA files in `MSA/` directory are used for evolutionary profiles

- [ ] **RNA-07** · Write clean commit message
  ```
  Rewrite: BiLSTM → Transformer encoder + MSA evolutionary features

  - Fix PAD masking bug (PAD=4 → PAD=0; mask_zero was silently broken)
  - Parse MSA/*.MSA.fasta files as position-frequency profiles (A/C/G/U/gap)
  - Pre-LN Transformer encoder (4 layers, 8 heads, GELU FFN)
  - Distance matrix consistency loss on C3' atom (rotation-invariant)
  - AdamW + cosine LR + gradient clipping + early stopping
  - Val RMSD reported in Angstroms every 5 epochs
  ```

- [ ] **RNA-08** · Push public repo to GitHub
  - Verify the repo is public and accessible without login

---

### Project 1 · LLM-Gene (parallel to RNA-01 while model trains)

**Goal**: Working Flask demo that answers bioinformatics questions from ClinVar + UniProt.

#### Environment setup

- [ ] **LG-01** · Install dependencies
  ```bash
  pip install sentence-transformers faiss-cpu rank-bm25 biopython flask
  # Install Ollama: https://ollama.ai
  ollama pull mistral
  ```

#### Knowledge base construction

- [ ] **LG-02** · Download ClinVar variant summary (~200 MB, public)
  ```bash
  wget https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz
  gunzip variant_summary.txt.gz
  # Keep columns: GeneSymbol, ClinicalSignificance, PhenotypeList, ReviewStatus
  ```

- [ ] **LG-03** · Download UniProt Swiss-Prot FASTA (~80 MB, public)
  ```bash
  wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
  gunzip uniprot_sprot.fasta.gz
  ```

- [ ] **LG-04** · Chunk documents
  - ClinVar rows → chunk by gene symbol (one chunk = all variants for one gene)
  - UniProt entries → chunk by protein (one chunk = function + keywords + organism)
  - Target: 500 tokens per chunk, 50-token overlap
  - Filter: English-language entries only; min 3 sentences per chunk

- [ ] **LG-05** · Build embeddings and FAISS index
  ```python
  from sentence_transformers import SentenceTransformer
  model = SentenceTransformer(
      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
  )
  embeddings = model.encode(chunks, batch_size=64, show_progress_bar=True)
  # Save: np.save("embeddings.npy", embeddings)

  import faiss
  index = faiss.IndexFlatIP(embeddings.shape[1])   # inner product = cosine on L2-normalised vecs
  faiss.normalize_L2(embeddings)
  index.add(embeddings)
  faiss.write_index(index, "bio_index.faiss")
  ```

#### Retrieval pipeline

- [ ] **LG-06** · Dense retrieval — top 20 by cosine similarity from FAISS
- [ ] **LG-07** · BM25 sparse retrieval
  ```python
  from rank_bm25 import BM25Okapi
  tokenized = [chunk.split() for chunk in chunks]
  bm25 = BM25Okapi(tokenized)
  scores = bm25.get_scores(query.split())
  ```
- [ ] **LG-08** · Reciprocal Rank Fusion of dense + BM25 results
  - `score_rrf(d_rank, s_rank) = 1/(k + d_rank) + 1/(k + s_rank)`, k=60
  - Take top 20 after fusion
- [ ] **LG-09** · Cross-encoder reranking
  ```python
  from sentence_transformers import CrossEncoder
  reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
  scores = reranker.predict([(query, passage) for passage in top_20])
  # Return top 5 by reranker score
  ```

#### Generation

- [ ] **LG-10** · Build Ollama generation wrapper
  ```python
  import subprocess, json

  def generate(query: str, context_passages: list[str]) -> str:
      context = "\n\n".join(f"[{i+1}] {p}" for i, p in enumerate(context_passages))
      prompt  = f"""You are a bioinformatics assistant. Answer using ONLY the
  provided context. Cite each claim with [source number].

  Context:
  {context}

  Question: {query}
  Answer:"""
      result = subprocess.run(
          ["ollama", "run", "mistral", prompt],
          capture_output=True, text=True
      )
      return result.stdout.strip()
  ```

- [ ] **LG-11** · Wire retrieval → generation into `pipeline.py`

#### Web interface

- [ ] **LG-12** · Flask app with single endpoint
  ```python
  @app.route("/gene-qa", methods=["POST"])
  def gene_qa():
      query    = request.json["query"]
      passages = retrieve(query)   # LG-06 to LG-09
      answer   = generate(query, passages)
      return jsonify({"answer": answer, "sources": passages[:5]})
  ```

- [ ] **LG-13** · Simple HTML frontend (copy Surbhi's BioNLP card style)

#### Evaluation

- [ ] **LG-14** · Compile 50 test Q&A pairs from ClinVar FAQ + UniProt help pages
- [ ] **LG-15** · Compute MRR on retrieval (does the correct passage appear in top-5?)
- [ ] **LG-16** · Manual grounding check on 20 random answers — is every claim sourced?

#### GitHub

- [ ] **LG-17** · Push public repo with README, demo GIF/screenshot, MRR score

---

## 🟡 Planned — Week 2–5

---

### Project 2 · BioMultiAgent

> Start after LLM-Gene demo is live. This is the highest-impact project for collaboration.

#### Phase 1 — Environment and fork (Day 1)

- [ ] **BMA-01** · Fork `Suru1496/Bio_NLP` on GitHub
- [ ] **BMA-02** · Install agent stack
  ```bash
  pip install langgraph langchain-community litellm chromadb ete3 biopython
  ollama pull mistral
  ```
- [ ] **BMA-03** · Verify BioPython works
  ```python
  from Bio import SeqIO, Entrez, pairwise2
  Entrez.email = "your@email.com"
  print("BioPython OK")
  ```

#### Phase 2 — State schema and supervisor (Days 2–4)

- [ ] **BMA-04** · Implement LangGraph state schema
  ```python
  from typing import TypedDict, List, Dict, Any

  class BioAgentState(TypedDict):
      query          : str
      intent         : str   # seq | align | annot | phylo | literature | compound
      sub_tasks      : List[str]
      agent_results  : Dict[str, Any]
      memory_context : List[Dict]
      final_response : str
      citations      : List[str]
  ```

- [ ] **BMA-05** · Implement intent classifier node
  - Prompt: "Classify this bioinformatics query. Output JSON: {intent, sub_tasks[]}"
  - Test on 20 sample queries (include ambiguous and compound cases)
  - Verify JSON parsing is robust (use `pydantic` or `json.loads` with try/except)

- [ ] **BMA-06** · Implement supervisor routing node (LangGraph conditional edge)
  ```python
  def route(state: BioAgentState) -> str:
      if state["intent"] == "compound":
          return "decompose"     # split into sub-tasks, call multiple agents
      return state["intent"]     # route directly to specialist agent
  ```

#### Phase 3 — Specialist agents (Days 3–8)

- [ ] **BMA-07** · `SeqAgent` — wrap existing BioNLP sequence analysis
  - Functions: `translate(seq)`, `gc_content(seq)`, `find_motifs(seq, pattern)`
  - Input: `{"sequence": str, "task": str}`
  - Output: `{"result": str, "details": dict}`

- [ ] **BMA-08** · `AlignAgent` — BioPython pairwise + MUSCLE
  - Pairwise: `pairwise2.align.globalxx(seq1, seq2)`
  - MSA: call MUSCLE subprocess for 3+ sequences
  - Output: aligned FASTA string + alignment score

- [ ] **BMA-09** · `AnnotAgent` — ORF prediction + GC content
  - Find all start/stop codon pairs in all 6 reading frames
  - Return: list of (start, stop, length, protein_sequence) per ORF

- [ ] **BMA-10** · `PhyloAgent` — NJ tree from aligned sequences
  ```python
  from Bio import Phylo
  from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
  # Returns: Newick string + ETE3 ASCII art render
  ```

- [ ] **BMA-11** · `LitAgent` — PubMed retrieval (Surbhi's explicit future scope)
  ```python
  from Bio import Entrez
  Entrez.email = "your@email.com"

  def search_pubmed(query: str, max_results: int = 5) -> List[Dict]:
      handle  = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
      record  = Entrez.read(handle)
      ids     = record["IdList"]
      handle2 = Entrez.efetch(db="pubmed", id=ids, rettype="abstract")
      # Parse and return title + abstract + PMID per result
  ```

#### Phase 4 — Memory layer (Days 6–9)

- [ ] **BMA-12** · Implement ChromaDB session store
  ```python
  import chromadb
  client     = chromadb.Client()
  collection = client.create_collection("bio_session_memory")

  def store_result(session_id: str, query: str, result: str) -> None:
      collection.add(
          documents=[f"Q: {query}\nA: {result}"],
          ids=[f"{session_id}_{int(time.time())}"],
          metadatas=[{"session_id": session_id}]
      )

  def retrieve_context(session_id: str, query: str, n: int = 3) -> List[str]:
      results = collection.query(query_texts=[query], n_results=n,
                                 where={"session_id": session_id})
      return results["documents"][0]
  ```

#### Phase 5 — Integration and demo (Days 10–14)

- [ ] **BMA-13** · Wire all agents into LangGraph graph
  - Nodes: supervisor → route → [seq/align/annot/phylo/lit] → fuse → respond
  - Edge: compound intent → loop over sub_tasks, collect results
  - Memory: inject session context into every supervisor prompt

- [ ] **BMA-14** · Replace BioNLP Platform's `/analyze` Flask endpoint
  ```python
  @app.route('/analyze', methods=['POST'])
  def analyze():
      query  = request.form['query']
      state  = run_bio_agent(query, session_id=session.get("id"))
      return jsonify({"result": state["final_response"],
                      "citations": state["citations"]})
  ```

- [ ] **BMA-15** · Test compound query end-to-end
  > "Translate this sequence, predict ORFs, align it to these 3 homologs,
  > build a phylogenetic tree, and find 5 recent PubMed papers."
  - All 5 agents must fire; result must be coherent; citations must appear

- [ ] **BMA-16** · Record 2-min screen demo (OBS or Loom)

- [ ] **BMA-17** · Push to GitHub fork with PR description referencing her report's future scope

---

### Project 3 · FedGenome

> Start Week 3, parallel to BioMultiAgent Phase 3.

#### Phase 1 — Data (Week 3)

- [ ] **FG-01** · Register GDC Data Portal: https://portal.gdc.cancer.gov
  - Download: TCGA-BRCA, TCGA-LUAD, TCGA-COAD — Masked Somatic Mutation (MAF) files
  - If access takes >1 week: use ClinVar as interim dataset (immediate, no registration)

- [ ] **FG-02** · Implement non-IID partition script
  ```python
  import numpy as np

  def dirichlet_partition(samples, n_sites=3, alpha=0.5, seed=42):
      """Dirichlet-based non-IID split, identical strategy to FedAlert."""
      rng    = np.random.default_rng(seed)
      labels = samples["cancer_type"].unique()
      sites  = [[] for _ in range(n_sites)]
      for label in labels:
          subset = samples[samples["cancer_type"] == label].index.tolist()
          props  = rng.dirichlet([alpha] * n_sites)
          splits = (props * len(subset)).astype(int)
          start  = 0
          for s_idx, count in enumerate(splits):
              sites[s_idx].extend(subset[start : start + count])
              start += count
      return sites
  ```

- [ ] **FG-03** · Build variant encoding pipeline
  - For each MAF row: extract ±10 bp genomic flanking context (use pysam or BioPython)
  - Tokenize as nucleotide sequence: ACGT → integer tokens
  - Label via ClinVar overlap: P/LP → pathogenic=1, B/LB → benign=0, VUS → skip

#### Phase 2 — Local model (Week 4)

- [ ] **FG-04** · Implement `LocalGenomeNet` (CNN-1D variant)
  ```python
  class LocalGenomeNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.encoder = nn.Sequential(
              nn.Conv1d(4, 64, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
              nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
              nn.AdaptiveAvgPool1d(1),
          )
          self.head = nn.Linear(128, 2)

      def forward(self, x):   # x: (B, 4, L) one-hot nucleotide
          return self.head(self.encoder(x).squeeze(-1))
  ```

- [ ] **FG-05** · Train `LocalGenomeNet` on single-site TCGA-BRCA
  - Target: AUC-ROC ≥ 0.75 before starting federation
  - If AUC < 0.65 → check label balance; apply class-weighted loss

#### Phase 3 — FL server (Week 5)

- [ ] **FG-06** · Install Flower
  ```bash
  pip install flwr[simulation]
  ```

- [ ] **FG-07** · Port FedAlert aggregator to Flower Strategy
  ```python
  import flwr as fl
  from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

  class FedGenomeStrategy(fl.server.strategy.FedAvg):
      def aggregate_fit(self, server_round, results, failures):
          # Extract per-client local precision scores from metrics
          precision_scores = [r.metrics.get("precision", 1.0) for _, r in results]
          weights_total    = sum(precision_scores)
          # Weighted aggregate: higher-precision clients contribute more
          agg_params = [
              np.sum([
                  (p / weights_total) * layer
                  for p, (_, fit_res) in zip(
                      precision_scores,
                      [(None, r) for _, r in results]
                  )
                  for layer in parameters_to_ndarrays(fit_res.parameters)[:1]
              ], axis=0)
          ]
          return ndarrays_to_parameters(agg_params), {}
  ```
  - Key: pass `precision` in `fit_res.metrics` from each Flower client

- [ ] **FG-08** · Implement 3 Flower clients (one per simulated site)
  - Each client: load site partition, train 1 local epoch, return model diff + precision metric

- [ ] **FG-09** · Run federated training: 20 communication rounds, 3 sites
  ```bash
  # Flower simulation mode (single machine, 3 virtual clients)
  python fedgenome_server.py --strategy fedalert --rounds 20
  ```

#### Phase 4 — Evaluation (Week 6–7)

- [ ] **FG-10** · Per-site AUC-ROC + global weighted AUC
- [ ] **FG-11** · Equity gap: `std(site_aucs)` — lower is better
- [ ] **FG-12** · Ablation table

  | Method | Global AUC | Site 1 AUC | Site 2 AUC | Site 3 AUC | Equity Gap |
  |--------|-----------|-----------|-----------|-----------|-----------|
  | Centralised | — | — | — | — | — |
  | FedAvg | — | — | — | — | — |
  | FedProx | — | — | — | — | — |
  | **FedGenome (ours)** | — | — | — | — | — |

- [ ] **FG-13** · Convergence plot: val AUC vs communication round for all 4 methods
- [ ] **FG-14** · Push code + results to public GitHub repo

---

## 📋 Backlog

### RNAStructFormer upgrades
- [ ] Replace learned embedding with frozen Nucleotide Transformer encoder (HuggingFace)
- [ ] Add base-pair interaction bias matrix to attention (riboswitch-aware)
- [ ] Multi-sequence training with reverse-complement data augmentation
- [ ] Frame-aligned point error (FAPE) loss — AlphaFold2's rotation-equivariant loss

### BioMultiAgent upgrades
- [ ] AlphaFold3 API integration as StructureAgent (protein structure prediction)
- [ ] BLAST+ integration in AlignAgent (remote NCBI BLAST API)
- [ ] Streaming response via WebSocket (show partial results as agents complete)
- [ ] Multi-session memory: persist ChromaDB across Flask restarts

### FedGenome upgrades
- [ ] Add Gaussian mechanism differential privacy (ε-DP guarantee per round)
- [ ] Cross-cancer generalisation: train on BRCA, evaluate on LUAD (zero-shot)
- [ ] Upgrade local encoder to Nucleotide Transformer (fine-tuned on TCGA data)
- [ ] Add client selection: only include clients with precision above threshold per round

### LLM-Gene upgrades
- [ ] Fine-tune Llama-3-8B on biomedical Q&A for domain-specific generation
- [ ] Add OMIM API integration (gene–disease relationship queries)
- [ ] PubMed full-text retrieval (not just abstracts) via PMC Open Access FTP
- [ ] Sequence classification sub-module (ESM-2 + pathogenicity head)

### All projects
- [ ] Add Weights & Biases experiment tracking: `wandb.init(project="bio-portfolio")`
- [ ] Write 1-page project summaries for each (use as LinkedIn post content)
- [ ] Record short demos for each project (YouTube unlisted, link in GitHub README)

---

## 🏁 Milestones

| ID | Milestone | Target | Success signal |
|----|-----------|--------|----------------|
| M0 | RNAStructFormer running | End of Week 1 | RMSD ≤ 6 Å logged, submission.csv generated, repo public |
| M1 | LLM-Gene demo live | End of Week 2 | Flask answers 5 test gene questions with citations |
| M2 | BioMultiAgent compound query | End of Week 5 | "Translate + align + find papers" runs in one query |
| M3 | FedGenome first FL results | End of Week 7 | 3-site convergence curve plotted, ablation table populated |
| **M4** | **Contact Surbhi Pawar** | **Week 6** | **LinkedIn message sent, 2+ repo links included** |
| M5 | First collaboration call | Week 8–10 | Meeting scheduled |
| M6 | Joint paper draft | Month 4–6 | Shared Overleaf document, agreed problem statement |

---

## 🔍 Discovery Log

> Things learned mid-process that changed direction or revealed new information.

| Date | Discovery | Impact |
|------|-----------|--------|
| 2026-06-06 | RNA project is Kaggle Stanford RNA 3D Folding competition dataset | MSA files are PDB-sourced alignments — already downloaded, significant preprocessing preserved |
| 2026-06-06 | Original PAD masking bug: PAD='P'→4 but `mask_zero=True` only guards index 0 | Every previous training run learned on padding noise — RMSD results were invalid |
| 2026-06-06 | MSA folder (fully downloaded) was never used in the original model | Biggest free improvement: just parsing the FASTAs adds co-evolutionary signal |
| 2026-06-06 | Surbhi's BioNLP Platform report explicitly lists "multi-agent AI systems" as future scope | BioMultiAgent directly addresses this gap using published DST-AgenticNet methodology |
| 2026-06-06 | Surbhi's GitHub has HER2 + BreastCancerPrediction repos — cancer genomics focus | FedGenome's TCGA-BRCA dataset directly overlaps with her existing cancer genomics work |
| 2026-06-06 | BioNLP report lists "limited external database connectivity" as a current limitation | LLM-Gene's ClinVar + UniProt retrieval layer is the exact fix she already identified as needed |
