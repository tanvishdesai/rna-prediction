"""End-to-end compound bioinformatics query demo."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioagent.supervisor import run_bio_agent

COMPOUND_QUERY = (
    "Translate this sequence ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG, "
    "predict ORFs, align it to homologs, build a phylogenetic tree, "
    "and find recent PubMed papers on this gene family."
)

DEMO_SEQ = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
HOMOLOGS = [
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA",
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCAT",
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGT",
]


def main() -> None:
    print("=" * 70)
    print("BioMultiAgent — Compound Query Demo")
    print("=" * 70)
    print(f"Query: {COMPOUND_QUERY}\n")

    state = run_bio_agent(
        COMPOUND_QUERY,
        sequence=DEMO_SEQ,
        homologs=HOMOLOGS,
    )

    print(state["final_response"])
    if state["citations"]:
        print("\nCitations:")
        for c in state["citations"]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
