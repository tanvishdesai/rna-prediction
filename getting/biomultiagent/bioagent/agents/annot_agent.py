"""AnnotAgent — ORF prediction across 6 reading frames."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from Bio.Seq import Seq

START_CODONS = {"ATG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}


def find_orfs(seq: str, min_len: int = 90) -> List[Dict]:
    dna = seq.upper().replace("U", "T")
    orfs: List[Dict] = []
    for frame in range(3):
        for strand, s in [(1, dna), (-1, str(Seq(dna).reverse_complement()))]:
            i = 0
            while i < len(s) - 2:
                codon = s[i : i + 3]
                if codon in START_CODONS:
                    j = i + 3
                    while j < len(s) - 2:
                        stop = s[j : j + 3]
                        if stop in STOP_CODONS:
                            length = j + 3 - i
                            if length >= min_len:
                                protein = str(Seq(s[i : j + 3]).translate())
                                orfs.append({
                                    "frame": frame,
                                    "strand": "+" if strand == 1 else "-",
                                    "start": i,
                                    "stop": j + 3,
                                    "length_nt": length,
                                    "protein": protein[:80],
                                })
                            i = j + 3
                            break
                        j += 3
                    else:
                        i += 3
                else:
                    i += 1
    return sorted(orfs, key=lambda x: -x["length_nt"])[:10]


def run(task: str, sequence: str = "", **kwargs) -> Dict[str, Any]:
    seq = sequence or kwargs.get("seq", "")
    if not seq:
        return {"result": "No sequence provided.", "orfs": []}
    orfs = find_orfs(seq)
    if not orfs:
        return {"result": "No ORFs ≥ 90 nt found.", "orfs": []}
    lines = [
        f"ORF {i+1}: frame={o['frame']} strand={o['strand']} len={o['length_nt']}nt protein={o['protein'][:40]}…"
        for i, o in enumerate(orfs[:5])
    ]
    return {"result": "\n".join(lines), "orfs": orfs}
