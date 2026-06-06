from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class BioAgentState(TypedDict):
    query: str
    intent: str
    sub_tasks: List[str]
    agent_results: Dict[str, Any]
    memory_context: List[str]
    final_response: str
    citations: List[str]
    session_id: str
