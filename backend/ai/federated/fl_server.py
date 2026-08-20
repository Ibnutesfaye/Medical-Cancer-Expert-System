"""
Federated Learning Server
==========================
Coordinates training across multiple hospitals using the Flower framework.
Each hospital trains locally on its own data and only sends model weights
(gradients) to the server — raw patient images never leave the hospital.

Architecture:
  Hospital A ──► gradients ──►
  Hospital B ──► gradients ──► FL Server (FedAvg) ──► Global Model
  Hospital C ──► gradients ──►

Usage:
    # Start server (run once)
    venv\\Scripts\\python.exe -m ai.federated.fl_server

    # Start clients in separate terminals (one per hospital)
    venv\\Scripts\\python.exe -m ai.federated.fl_client --hospital A
    venv\\Scripts\\python.exe -m ai.federated.fl_client --hospital B
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

BASE = Path(__file__).parent.parent.parent


def start_fl_server(
    num_rounds:          int = 20,
    min_clients:         int = 2,
    server_address:      str = "0.0.0.0:8080",
    output_model_path:   Optional[str] = None,
):
    """
    Start the Federated Learning server.

    Args:
        num_rounds:       Number of federated rounds
        min_clients:      Minimum hospitals required per round
        server_address:   gRPC server address
        output_model_path: Where to save the final global model
    """
    try:
        import flwr as fl
    except ImportError:
        print("Flower not installed. Run: pip install flwr==1.8.0")
        return

    import torch
    import torchvision.models as models
    import torch.nn as nn

    # Load label info to get num_classes
    labels_path = BASE / "label_classes.json"
    with open(labels_path) as f:
        label_info = json.load(f)
    num_classes = len(label_info["class_to_idx"])

    # Create initial model
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    initial_params = [p.detach().numpy() for p in model.parameters()]

    def weighted_average(metrics):
        """Aggregate accuracy metrics from all clients."""
        accuracies = [num * m["accuracy"] for num, m in metrics]
        totals     = [num for num, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(totals)}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit              = 1.0,
        fraction_evaluate         = 1.0,
        min_fit_clients           = min_clients,
        min_evaluate_clients      = min_clients,
        min_available_clients     = min_clients,
        initial_parameters        = fl.common.ndarrays_to_parameters(initial_params),
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    print(f"Starting FL server on {server_address}")
    print(f"  Rounds     : {num_rounds}")
    print(f"  Min clients: {min_clients}")

    fl.server.start_server(
        server_address = server_address,
        config         = fl.server.ServerConfig(num_rounds=num_rounds),
        strategy       = strategy,
    )

    print("Federated training complete.")


if __name__ == "__main__":
    start_fl_server()
