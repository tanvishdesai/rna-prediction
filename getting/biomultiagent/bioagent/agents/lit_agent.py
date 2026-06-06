"""LitAgent — PubMed literature search via NCBI Entrez."""

from __future__ import annotations

import os
from typing import Any, Dict, List

from Bio import Entrez

Entrez.email = os.environ.get("ENTREZ_EMAIL", "bioagent@example.com")


def search_pubmed(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
    record = Entrez.read(handle)
    ids = record.get("IdList", [])
    if not ids:
        return []

    handle2 = Entrez.efetch(db="pubmed", id=ids, rettype="abstract", retmode="xml")
    papers = Entrez.read(handle2)
    results = []
    for article in papers.get("PubmedArticle", []):
        med = article["MedlineCitation"]["Article"]
        pmid = str(article["MedlineCitation"]["PMID"])
        title = str(med.get("ArticleTitle", ""))
        abstract_parts = med.get("Abstract", {}).get("AbstractText", [])
        if not isinstance(abstract_parts, list):
            abstract_parts = [abstract_parts]
        abstract = " ".join(str(p) for p in abstract_parts)[:500]
        results.append({"pmid": pmid, "title": title, "abstract": abstract})
    return results


def run(task: str, query: str = "", **kwargs) -> Dict[str, Any]:
    q = query or kwargs.get("pubmed_query", task)
    papers = search_pubmed(q, max_results=kwargs.get("max_results", 5))
    if not papers:
        return {"result": f"No PubMed results for: {q}", "papers": [], "citations": []}
    lines = [f"[PMID:{p['pmid']}] {p['title']}" for p in papers]
    citations = [f"https://pubmed.ncbi.nlm.nih.gov/{p['pmid']}/" for p in papers]
    return {
        "result": "\n".join(lines),
        "papers": papers,
        "citations": citations,
    }
