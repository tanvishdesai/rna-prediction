"""Compute MRR on gold Q&A pairs for retrieval evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import get_pipeline

GOLD_PATH = Path(__file__).parent / "gold_qa.json"


def mrr(pipeline, gold: list) -> float:
    scores = []
    for item in gold:
        query = item["query"]
        expected_genes = set(item.get("expected_genes", []))
        passages = pipeline.retrieve(query, top_k=5)
        rank = 0
        for i, p in enumerate(passages, 1):
            if p.get("gene", "").upper() in {g.upper() for g in expected_genes}:
                rank = i
                break
        scores.append(1.0 / rank if rank else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def main() -> None:
    if not GOLD_PATH.exists():
        print("gold_qa.json not found — skipping evaluation.")
        return
    with GOLD_PATH.open(encoding="utf-8") as fh:
        gold = json.load(fh)
    pipeline = get_pipeline()
    score = mrr(pipeline, gold)
    print(f"MRR@5: {score:.3f}  ({len(gold)} queries)")


if __name__ == "__main__":
    main()
