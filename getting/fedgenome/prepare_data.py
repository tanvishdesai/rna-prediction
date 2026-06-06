"""Build variant dataset with simulated multi-site non-IID partitions."""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
CONTEXT_LEN = 41
SEED = 42

# Simulate 3 hospital sites by chromosome bands (non-IID by genomic region)
SITE_CHROMS = {
    "site_1": {"1", "2", "3", "4", "5", "6", "7", "8"},      # early chromosomes
    "site_2": {"9", "10", "11", "12", "13", "14", "15", "16"},
    "site_3": {"17", "18", "19", "20", "21", "22", "X", "Y"},
}

PATHOGENIC = {"Pathogenic", "Likely pathogenic"}
BENIGN = {"Benign", "Likely benign"}


def _make_context(name: str, chrom: str) -> str:
    """Synthesise ±20 bp context from variant name + chromosome."""
    seed_val = hash(f"{chrom}_{name}") % (2**32)
    rng = random.Random(seed_val)
    bases = "ACGT"
    return "".join(rng.choice(bases) for _ in range(CONTEXT_LEN))


def load_variants(path: Path, max_per_class: int = 8000) -> pd.DataFrame:
    usecols = ["Name", "Chromosome", "ClinicalSignificance", "GeneSymbol", "Type"]
    df = pd.read_csv(path, sep="\t", usecols=lambda c: c in usecols, low_memory=False)
    df["Chromosome"] = df["Chromosome"].astype(str).str.replace("chr", "", regex=False)
    df["label"] = df["ClinicalSignificance"].apply(
        lambda s: 1 if any(p in str(s) for p in PATHOGENIC)
        else (0 if any(b in str(s) for b in BENIGN) else -1)
    )
    df = df[df["label"] >= 0]
    df = df[df["Type"].astype(str).str.contains("single nucleotide", case=False, na=False)]

    pos = df[df["label"] == 1].head(max_per_class)
    neg = df[df["label"] == 0].head(max_per_class)
    df = pd.concat([pos, neg], ignore_index=True)
    df["context"] = df.apply(lambda r: _make_context(str(r["Name"]), str(r["Chromosome"])), axis=1)
    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def assign_sites(df: pd.DataFrame) -> pd.DataFrame:
    def _site(chrom: str) -> str:
        for site, chrs in SITE_CHROMS.items():
            if chrom in chrs:
                return site
        return "site_1"
    df["site"] = df["Chromosome"].apply(_site)
    return df


def save_partitions(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    records = df[["context", "label", "site", "GeneSymbol", "Chromosome"]].to_dict(orient="records")
    with (DATA_DIR / "variants.json").open("w") as fh:
        json.dump(records, fh)
    for site in SITE_CHROMS:
        subset = [r for r in records if r["site"] == site]
        with (DATA_DIR / f"{site}.json").open("w") as fh:
            json.dump(subset, fh)
        print(f"  {site}: {len(subset)} variants")


def main() -> None:
    path = DATA_DIR / "variant_summary.txt"
    if not path.exists():
        raise FileNotFoundError("Run: python scripts/download_data.py first")
    print("Building variant dataset …")
    df = load_variants(path)
    df = assign_sites(df)
    save_partitions(df)
    print(f"Total: {len(df)} variants → {DATA_DIR}")


if __name__ == "__main__":
    main()
