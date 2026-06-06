"""LangGraph supervisor orchestrating specialist bioinformatics agents."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any, Dict, List

from langgraph.graph import END, StateGraph

from bioagent.agents import AGENTS
from bioagent.memory.chroma_store import SessionMemory
from bioagent.state import BioAgentState

# Demo sequence and homologs for compound queries
DEMO_SEQ = "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGATAG"
DEMO_HOMOLOGS = [
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGA",
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCAT",
    "ATGGCCATTGTAATGGGCCGCTGAAAGGGTGCCCGT",
]

INTENT_KEYWORDS = {
    "seq": ["translate", "gc content", "gc%", "motif", "sequence analysis"],
    "align": ["align", "alignment", "msa", "multiple sequence"],
    "annot": ["orf", "annotate", "annotation", "open reading frame", "codon"],
    "phylo": ["phylo", "tree", "phylogen", "neighbor", "evolution"],
    "literature": ["pubmed", "paper", "literature", "article", "recent"],
}


def _classify_intent(query: str) -> tuple[str, list[str]]:
    q = query.lower()
    matched = [k for k, kws in INTENT_KEYWORDS.items() if any(kw in q for kw in kws)]
    if len(matched) >= 2 or " and " in q:
        sub_tasks = []
        for intent in ["seq", "annot", "align", "phylo", "literature"]:
            if intent in matched or (intent == "seq" and "translate" in q):
                sub_tasks.append(intent)
        if not sub_tasks:
            sub_tasks = ["seq", "annot", "align", "phylo", "literature"]
        return "compound", sub_tasks
    if matched:
        return matched[0], [matched[0]]
    if "translate" in q:
        return "seq", ["seq"]
    return "seq", ["seq"]


def _extract_sequence(query: str) -> str:
    m = re.search(r"[ATGCUatgcu]{20,}", query)
    return m.group(0).upper() if m else DEMO_SEQ


def _run_agent(intent: str, query: str, state: BioAgentState) -> Dict[str, Any]:
    seq = _extract_sequence(query)
    if intent == "seq":
        return AGENTS["seq"](query, sequence=seq)
    if intent == "align":
        return AGENTS["align"](query, sequences=[seq] + DEMO_HOMOLOGS)
    if intent == "annot":
        return AGENTS["annot"](query, sequence=seq)
    if intent == "phylo":
        return AGENTS["phylo"](query, sequences=[seq] + DEMO_HOMOLOGS)
    if intent == "literature":
        gene_match = re.search(r"\b([A-Z][A-Z0-9]{1,9})\b", query)
        term = gene_match.group(1) if gene_match else "gene family"
        return AGENTS["literature"](query, query=f"{term} gene family")
    return {"result": f"Unknown intent: {intent}"}


def classify_node(state: BioAgentState) -> BioAgentState:
    intent, sub_tasks = _classify_intent(state["query"])
    memory = SessionMemory()
    ctx = memory.retrieve(state.get("session_id", "default"), state["query"])
    return {
        **state,
        "intent": intent,
        "sub_tasks": sub_tasks,
        "memory_context": ctx,
        "agent_results": {},
        "citations": [],
    }


def execute_node(state: BioAgentState) -> BioAgentState:
    results: Dict[str, Any] = {}
    citations: List[str] = []
    for task in state["sub_tasks"]:
        out = _run_agent(task, state["query"], state)
        results[task] = out
        citations.extend(out.get("citations", []))
    return {**state, "agent_results": results, "citations": citations}


def fuse_node(state: BioAgentState) -> BioAgentState:
    parts = []
    for task, out in state["agent_results"].items():
        parts.append(f"## {task.upper()} Agent\n{out.get('result', '')}")
    response = "\n\n".join(parts)
    if state["memory_context"]:
        response = "Previous context:\n" + "\n".join(state["memory_context"][:2]) + "\n\n" + response
    memory = SessionMemory()
    memory.store(state.get("session_id", "default"), state["query"], response[:500])
    return {**state, "final_response": response}


def build_graph():
    g = StateGraph(BioAgentState)
    g.add_node("classify", classify_node)
    g.add_node("execute", execute_node)
    g.add_node("fuse", fuse_node)
    g.set_entry_point("classify")
    g.add_edge("classify", "execute")
    g.add_edge("execute", "fuse")
    g.add_edge("fuse", END)
    return g.compile()


_graph = None


def run_bio_agent(
    query: str,
    session_id: str | None = None,
    sequence: str | None = None,
    homologs: List[str] | None = None,
) -> BioAgentState:
    global DEMO_SEQ, DEMO_HOMOLOGS
    if sequence:
        DEMO_SEQ = sequence
    if homologs:
        DEMO_HOMOLOGS = homologs

    global _graph
    if _graph is None:
        _graph = build_graph()

    init: BioAgentState = {
        "query": query,
        "intent": "",
        "sub_tasks": [],
        "agent_results": {},
        "memory_context": [],
        "final_response": "",
        "citations": [],
        "session_id": session_id or str(uuid.uuid4()),
    }
    return _graph.invoke(init)
