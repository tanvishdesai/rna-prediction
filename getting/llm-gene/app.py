"""Flask web API for LLM-Gene bioinformatics Q&A."""

from __future__ import annotations

from flask import Flask, jsonify, render_template, request

from pipeline import get_pipeline

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/gene-qa", methods=["POST"])
def gene_qa():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    if not query:
        return jsonify({"error": "query is required"}), 400
    result = get_pipeline().answer(query)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
