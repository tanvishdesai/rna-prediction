"""Flask API compatible with BioNLP Platform /analyze endpoint."""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flask import Flask, jsonify, render_template, request, session

from bioagent.supervisor import run_bio_agent

app = Flask(__name__)
app.secret_key = "biomultiagent-dev-key"


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    query = request.form.get("query") or (request.json or {}).get("query", "")
    if not query:
        return jsonify({"error": "query required"}), 400
    sid = session.get("id") or str(uuid.uuid4())
    session["id"] = sid
    state = run_bio_agent(query, session_id=sid)
    return jsonify({
        "result": state["final_response"],
        "citations": state["citations"],
        "intent": state["intent"],
        "agents_used": state["sub_tasks"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
