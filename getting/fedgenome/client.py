"""Flower client for FedGenome local training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.client import Client, ClientApp
from flwr.common import Context, Metrics, NDArrays, Scalar
from sklearn.metrics import precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

from model import LocalGenomeNet, one_hot_encode

DATA_DIR = Path(__file__).parent / "data"
BATCH_SIZE = 64
LOCAL_EPOCHS = 2
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class VariantDataset(Dataset):
    def __init__(self, records: list) -> None:
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        r = self.records[idx]
        x = one_hot_encode(r["context"])
        y = int(r["label"])
        return x, y


def get_parameters(model: nn.Module) -> NDArrays:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: NDArrays) -> None:
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {k: torch.tensor(v) for k, v in params_dict}
    model.load_state_dict(state_dict, strict=True)


def train_local(
    model: nn.Module,
    loader: DataLoader,
    epochs: int,
    class_weights: torch.Tensor | None = None,
) -> Tuple[float, float]:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
    return _evaluate(model, loader)


def _evaluate(model: nn.Module, loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    ys, ps, prs = [], [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            pred = logits.argmax(1).cpu().numpy()
            ys.extend(yb.numpy())
            ps.extend(prob)
            prs.extend(pred)
    try:
        auc = roc_auc_score(ys, ps)
    except ValueError:
        auc = 0.5
    prec = precision_score(ys, prs, zero_division=0)
    return auc, prec


class FedGenomeClient(Client):
    def __init__(self, site: str, records: list) -> None:
        self.site = site
        self.records = records
        self.model = LocalGenomeNet().to(DEVICE)
        labels = [r["label"] for r in records]
        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        w = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)
        self.class_weights = w
        self.train_ds = VariantDataset(records)
        self.train_loader = DataLoader(self.train_ds, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        auc, prec = train_local(
            self.model, self.train_loader, LOCAL_EPOCHS, self.class_weights
        )
        return get_parameters(self.model), len(self.records), {
            "auc": auc, "precision": prec, "site": self.site,
        }

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        auc, prec = _evaluate(self.model, self.train_loader)
        return float(1.0 - auc), len(self.records), {"auc": auc, "precision": prec}


def client_fn(context: Context):
    partition_id = int(context.node_config["partition-id"])
    sites = ["site_1", "site_2", "site_3"]
    site = sites[partition_id % 3]
    path = DATA_DIR / f"{site}.json"
    with path.open() as fh:
        records = json.load(fh)
    return FedGenomeClient(site, records).to_client()


app = ClientApp(client_fn=client_fn)
