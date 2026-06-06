"""PhyloAgent — Neighbor-Joining tree from aligned sequences."""

from __future__ import annotations

import io
from typing import Any, Dict, List

from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment


def build_nj_tree(sequences: List[str], names: List[str] | None = None) -> Dict[str, Any]:
    if len(sequences) < 3:
        return {"result": "Need ≥3 sequences for phylogenetic tree.", "newick": ""}

    labels = names or [f"seq{i+1}" for i in range(len(sequences))]
    records = [
        SeqRecord(Seq(s.upper().replace("U", "T")), id=labels[i], description="")
        for i, s in enumerate(sequences)
    ]
    # Pad to equal length for distance matrix
    max_len = max(len(r.seq) for r in records)
    for r in records:
        r.seq = r.seq + "N" * (max_len - len(r.seq))

    aln = MultipleSeqAlignment(records)
    calculator = DistanceCalculator("identity")
    dm = calculator.get_distance(aln)
    constructor = DistanceTreeConstructor()
    tree = constructor.nj(dm)

    buf = io.StringIO()
    Phylo.write(tree, buf, "newick")
    newick = buf.getvalue().strip()

    ascii_buf = io.StringIO()
    Phylo.draw_ascii(tree, file=ascii_buf)
    ascii_tree = ascii_buf.getvalue()

    return {
        "result": f"NJ tree (Newick):\n{newick}\n\nASCII:\n{ascii_tree}",
        "newick": newick,
        "ascii": ascii_tree,
    }


def run(task: str, sequences: List[str] | None = None, **kwargs) -> Dict[str, Any]:
    seqs = sequences or kwargs.get("seqs", [])
    return build_nj_tree(seqs, kwargs.get("names"))
