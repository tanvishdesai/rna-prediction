"""SeqAgent — sequence translation, GC content, motif search."""

from __future__ import annotations

import re
from typing import Any, Dict

from Bio.Seq import Seq


def gc_content(seq: str) -> float:
    s = seq.upper().replace("U", "T")
    if not s:
        return 0.0
    gc = sum(1 for c in s if c in "GC")
    return round(100.0 * gc / len(s), 2)


def translate(seq: str, frame: int = 0) -> str:
    dna = seq.upper().replace("U", "T")
    return str(Seq(dna[frame:]).translate(to_stop=True))


def find_motifs(seq: str, pattern: str) -> list:
    return [m.start() for m in re.finditer(pattern, seq.upper())]


def run(task: str, sequence: str = "", **kwargs) -> Dict[str, Any]:
    seq = sequence or kwargs.get("seq", "")
    if not seq:
        return {"result": "No sequence provided.", "details": {}}

    task_l = task.lower()
    details: Dict[str, Any] = {"length": len(seq)}

    if "gc" in task_l or "content" in task_l:
        details["gc_percent"] = gc_content(seq)
        return {"result": f"GC content: {details['gc_percent']}%", "details": details}

    if "translate" in task_l or "protein" in task_l:
        protein = translate(seq)
        details["protein"] = protein[:200]
        return {"result": f"Translation (frame 0): {protein[:120]}…", "details": details}

    if "motif" in task_l:
        pattern = kwargs.get("pattern", "ATG")
        positions = find_motifs(seq, pattern)
        details["positions"] = positions
        return {"result": f"Motif '{pattern}' at positions: {positions}", "details": details}

    details["gc_percent"] = gc_content(seq)
    return {
        "result": f"Sequence length {len(seq)}, GC={details['gc_percent']}%",
        "details": details,
    }
