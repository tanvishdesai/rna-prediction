"""Run FedGenome federated learning simulation (3 sites, manual + Flower-compatible aggregation)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset

from client import (
    VariantDataset,
    _evaluate,
    get_parameters,
    set_parameters,
    train_local,
)
from model import LocalGenomeNet
from strategy import _weighted_average

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
SITES = ["site_1", "site_2", "site_3"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64


def load_site_records(site: str) -> list:
    with (DATA_DIR / f"{site}.json").open(encoding="utf-8") as fh:
        return json.load(fh)


def site_loader(site: str, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        VariantDataset(load_site_records(site)),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )


def evaluate_all_sites(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    results = {}
    for site in SITES:
        loader = site_loader(site, shuffle=False)
        auc, prec = _evaluate(model, loader)
        results[site] = {"auc": round(auc, 4), "precision": round(prec, 4)}
    return results


def run_centralized(epochs: int = 10) -> Dict:
    all_records = []
    for site in SITES:
        all_records.extend(load_site_records(site))
    model = LocalGenomeNet().to(DEVICE)
    loader = DataLoader(VariantDataset(all_records), batch_size=BATCH_SIZE, shuffle=True)
    train_local(model, loader, epochs=epochs)
    per_site = evaluate_all_sites(model)
    aucs = [v["auc"] for v in per_site.values()]
    return {
        "per_site": per_site,
        "global_auc": round(float(np.mean(aucs)), 4),
        "equity_gap": round(float(np.std(aucs)), 4),
    }


def run_federated(strategy: str, rounds: int, local_epochs: int = 2) -> Dict:
    """
    Simulate 3 hospital sites training a shared LocalGenomeNet.

    strategy:
      fedavg    — weight clients by dataset size
      fedgenome — precision-weighted aggregation (FedAlert-style)
    """
    global_model = LocalGenomeNet().to(DEVICE)
    site_records = {s: load_site_records(s) for s in SITES}
    history: List[float] = []

    for rnd in range(1, rounds + 1):
        local_params: List = []
        weights: List[float] = []

        for site in SITES:
            local_model = LocalGenomeNet().to(DEVICE)
            set_parameters(local_model, get_parameters(global_model))
            loader = DataLoader(
                VariantDataset(site_records[site]),
                batch_size=BATCH_SIZE,
                shuffle=True,
            )
            _, prec = train_local(local_model, loader, epochs=local_epochs)
            local_params.append(get_parameters(local_model))
            if strategy == "fedgenome":
                weights.append(max(float(prec), 0.01))
            else:
                weights.append(float(len(site_records[site])))

        aggregated = _weighted_average(local_params, weights)
        set_parameters(global_model, aggregated)

        per_site = evaluate_all_sites(global_model)
        mean_auc = float(np.mean([v["auc"] for v in per_site.values()]))
        history.append(mean_auc)
        print(f"  round {rnd:2d}/{rounds}  mean AUC={mean_auc:.4f}")

    per_site = evaluate_all_sites(global_model)
    aucs = [v["auc"] for v in per_site.values()]
    return {
        "per_site": per_site,
        "global_auc": round(float(np.mean(aucs)), 4),
        "equity_gap": round(float(np.std(aucs)), 4),
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        default="all",
        choices=["all", "central", "fedavg", "fedgenome"],
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=2)
    args = parser.parse_args()

    if not (DATA_DIR / "variants.json").exists():
        raise FileNotFoundError("Run: python scripts/download_data.py && python prepare_data.py")

    RESULTS_DIR.mkdir(exist_ok=True)
    summary: Dict = {}

    if args.strategy in ("all", "central"):
        print("Training centralized baseline …")
        summary["centralized"] = run_centralized(epochs=args.rounds)
        print(
            f"  Centralized global AUC: {summary['centralized']['global_auc']:.4f}  "
            f"equity_gap: {summary['centralized']['equity_gap']:.4f}"
        )

    for strat in ["fedavg", "fedgenome"]:
        if args.strategy not in ("all", strat):
            continue
        print(f"\nRunning FL: {strat} ({args.rounds} rounds) …")
        summary[strat] = run_federated(strat, args.rounds, args.local_epochs)
        print(
            f"  {strat} global AUC: {summary[strat]['global_auc']:.4f}  "
            f"equity_gap: {summary[strat]['equity_gap']:.4f}"
        )

    out_path = RESULTS_DIR / "ablation.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nResults saved → {out_path}")

    labels, aucs = [], []
    for key, val in summary.items():
        labels.append(key)
        aucs.append(val.get("global_auc", 0.0))
    plt.figure(figsize=(6, 4))
    plt.bar(labels, aucs, color=["#2e86c1", "#28b463", "#e74c3c"][: len(labels)])
    plt.ylabel("AUC-ROC")
    plt.title("FedGenome Ablation")
    plt.ylim(0, 1)
    plt.tight_layout()
    plot_path = RESULTS_DIR / "ablation.png"
    plt.savefig(plot_path)
    print(f"Plot saved → {plot_path}")


if __name__ == "__main__":
    main()
