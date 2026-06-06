"""Download ClinVar variant_summary for FedGenome training."""

from __future__ import annotations

import gzip
import shutil
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gz_path = DATA_DIR / "variant_summary.txt.gz"
    out_path = DATA_DIR / "variant_summary.txt"

    if not out_path.exists():
        print("Downloading ClinVar variant_summary …")
        urllib.request.urlretrieve(URL, gz_path)
        with gzip.open(gz_path, "rb") as fin, out_path.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"Saved → {out_path}")
    else:
        print(f"Exists: {out_path}")

    print("Run: python prepare_data.py")


if __name__ == "__main__":
    main()
