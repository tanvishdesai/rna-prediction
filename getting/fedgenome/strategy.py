"""FedGenome aggregation strategies: FedAvg, FedProx-style, FedAlert precision-weighted."""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
from flwr.common import Metrics, NDArrays, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg


def _weighted_average(params_list: List[NDArrays], weights: List[float]) -> NDArrays:
    total = sum(weights)
    w = [x / total for x in weights]
    avg = []
    for layer_params in zip(*params_list):
        avg.append(sum(wi * p for wi, p in zip(w, layer_params)))
    return avg


class FedGenomeStrategy(FedAvg):
    """Precision-weighted FedAvg — higher-precision clients contribute more."""

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}
        params_list = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        precisions = [
            fit_res.metrics.get("precision", fit_res.metrics.get("val_precision", 0.5))
            for _, fit_res in results
        ]
        precisions = [max(float(p), 0.01) for p in precisions]
        aggregated = _weighted_average(params_list, precisions)
        from flwr.common import ndarrays_to_parameters
        return ndarrays_to_parameters(aggregated), {
            "fedgenome_precision_weights": str(precisions),
        }


class FedProxStrategy(FedAvg):
    """Standard FedAvg (FedProx local training handled in client)."""

    pass
