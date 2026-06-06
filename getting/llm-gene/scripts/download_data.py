"""Download ClinVar variant summary and UniProt Swiss-Prot for index building."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SOURCES = {
    "variant_summary.txt.gz": (
        "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
    ),
    "uniprot_sprot.fasta.gz": (
        "https://ftp.uniprot.org/pub/databases/uniprot/current_release/"
        "knowledgebase/complete/uniprot_sprot.fasta.gz"
    ),
}


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  downloading: {dest.name}")
    urllib.request.urlretrieve(url, dest)


def gunzip(src: Path) -> Path:
    out = src.with_suffix("")
    if out.exists():
        return out
    print(f"  extracting: {src.name}")
    with gzip.open(src, "rb") as fin, out.open("wb") as fout:
        shutil.copyfileobj(fin, fout)
    return out


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Data directory: {DATA_DIR}")
    for name, url in SOURCES.items():
        gz = DATA_DIR / name
        download(url, gz)
        gunzip(gz)
    print("Done. Run: python build_index.py")


if __name__ == "__main__":
    main()
