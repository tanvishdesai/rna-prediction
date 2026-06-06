"""AlignAgent — pairwise alignment and simple MSA."""

from __future__ import annotations

from typing import Any, Dict, List

from Bio import pairwise2
from Bio.Seq import Seq


def pairwise_align(seq1: str, seq2: str) -> Dict[str, Any]:
    s1, s2 = seq1.upper().replace("U", "T"), seq2.upper().replace("U", "T")
    alns = pairwise2.align.globalxx(s1, s2)
    if not alns:
        return {"result": "No alignment found.", "score": 0}
    best = alns[0]
    return {
        "result": f"Alignment score={best[2]}\n{best[0][:60]}…\n{best[1][:60]}…",
        "score": best[2],
        "aligned_a": str(best[0]),
        "aligned_b": str(best[1]),
    }


def simple_msa(sequences: List[str]) -> str:
    """Progressive star MSA (no external MUSCLE required)."""
    if len(sequences) < 2:
        return sequences[0] if sequences else ""
    ref = sequences[0].upper().replace("U", "T")
    aligned = [ref]
    for seq in sequences[1:]:
        result = pairwise_align(ref, seq)
        aligned.append(result.get("aligned_b", seq))
    max_len = max(len(s) for s in aligned)
    return "\n".join(s + "-" * (max_len - len(s)) for s in aligned)


def run(task: str, sequences: List[str] | None = None, **kwargs) -> Dict[str, Any]:
    seqs = sequences or kwargs.get("seqs", [])
    if len(seqs) >= 2 and ("msa" in task.lower() or "multiple" in task.lower() or len(seqs) > 2):
        msa = simple_msa(seqs)
        return {"result": f"MSA ({len(seqs)} sequences):\n{msa[:500]}", "msa": msa}
    if len(seqs) >= 2:
        return pairwise_align(seqs[0], seqs[1])
    seq = kwargs.get("sequence", "")
    return {"result": "Provide at least 2 sequences for alignment.", "details": {}}
