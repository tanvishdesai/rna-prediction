"""Chunk ClinVar + UniProt, embed with PubMedBERT, build FAISS index."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np
import pandas as pd
from Bio import SeqIO
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).parent / "data"
INDEX_DIR = DATA_DIR / "index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBED_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"


def _split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    chunks, start = [], 0
    while start < len(words):
        chunk = " ".join(words[start : start + size])
        if len(chunk.split(".")) >= 3 or len(words) <= size:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def load_clinvar_chunks(path: Path) -> List[Dict]:
    print(f"Loading ClinVar from {path.name} …")
    usecols = [
        "GeneSymbol", "ClinicalSignificance", "PhenotypeList",
        "ReviewStatus", "Name", "VariationID",
    ]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols, low_memory=False)
    df = df.dropna(subset=["GeneSymbol"])
    df = df[df["GeneSymbol"].str.len() > 0]

    chunks: List[Dict] = []
    for gene, grp in df.groupby("GeneSymbol"):
        lines = []
        for _, row in grp.head(200).iterrows():
            sig = str(row.get("ClinicalSignificance", ""))
            phen = str(row.get("PhenotypeList", ""))[:200]
            name = str(row.get("Name", ""))
            lines.append(f"Variant {name}: significance={sig}; phenotypes={phen}")
        text = f"Gene {gene}. " + " ".join(lines)
        for i, chunk in enumerate(_split_text(text)):
            chunks.append({
                "text": chunk,
                "source": "ClinVar",
                "gene": gene,
                "id": f"clinvar_{gene}_{i}",
            })
    print(f"  ClinVar chunks: {len(chunks)}")
    return chunks


def load_uniprot_chunks(path: Path, max_entries: int = 5000) -> List[Dict]:
    print(f"Loading UniProt from {path.name} …")
    chunks: List[Dict] = []
    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        if i >= max_entries:
            break
        desc = record.description
        organism = ""
        if "OS=" in desc:
            organism = desc.split("OS=")[1].split("OX=")[0].strip()
        gene = record.id
        func_match = re.search(r" ([A-Z][A-Za-z0-9-]+) ", desc)
        if func_match:
            gene = func_match.group(1)
        text = (
            f"Protein {record.id} ({gene}). Organism: {organism}. "
            f"Sequence length: {len(record.seq)}. Description: {desc[:400]}"
        )
        for j, chunk in enumerate(_split_text(text)):
            chunks.append({
                "text": chunk,
                "source": "UniProt",
                "gene": gene,
                "id": f"uniprot_{record.id}_{j}",
            })
    print(f"  UniProt chunks: {len(chunks)}")
    return chunks


def build_index(chunks: List[Dict], model: SentenceTransformer) -> Tuple[faiss.Index, np.ndarray]:
    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks …")
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index, embeddings


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    clinvar_path = DATA_DIR / "variant_summary.txt"
    uniprot_path = DATA_DIR / "uniprot_sprot.fasta"

    if not clinvar_path.exists() or not uniprot_path.exists():
        raise FileNotFoundError(
            "Missing data files. Run: python scripts/download_data.py"
        )

    chunks = load_clinvar_chunks(clinvar_path) + load_uniprot_chunks(uniprot_path)
    model = SentenceTransformer(EMBED_MODEL)
    index, _ = build_index(chunks, model)

    faiss.write_index(index, str(INDEX_DIR / "bio_index.faiss"))
    with (INDEX_DIR / "chunks.json").open("w", encoding="utf-8") as fh:
        json.dump(chunks, fh, ensure_ascii=False)
    print(f"Index saved → {INDEX_DIR}")


if __name__ == "__main__":
    main()
