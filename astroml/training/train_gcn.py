import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from astroml.models.gcn import GCN
from astroml.training.metrics import (
    TRAINING_EPOCHS_TOTAL,
    TRAINING_LOSS,
    TRAINING_ACCURACY,
    TRAINING_DURATION,
    MODEL_PARAMETERS,
    LEARNING_RATE,
)
from astroml.training.metrics_server import start_metrics_server


def train():
    # Start Prometheus metrics server
    start_metrics_server()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dim=16,
        output_dim=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    MODEL_PARAMETERS.labels(model_type="gcn").set(total_params)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    LEARNING_RATE.labels(model_type="gcn").set(0.01)

    for epoch in range(1, 201):
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        # Update training metrics
        TRAINING_EPOCHS_TOTAL.labels(model_type="gcn", dataset="cora").inc()
        TRAINING_LOSS.labels(model_type="gcn", dataset="cora", phase="train").set(loss.item())

        if epoch % 20 == 0:
            val_acc = _accuracy(model, data, data.val_mask)
            TRAINING_ACCURACY.labels(model_type="gcn", dataset="cora", phase="val").set(val_acc)
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f} | Val Acc: {val_acc:.4f}")

        # Log epoch duration
        epoch_duration = time.time() - epoch_start
        TRAINING_DURATION.labels(model_type="gcn", dataset="cora").observe(epoch_duration)

    test_acc = _accuracy(model, data, data.test_mask)
    TRAINING_ACCURACY.labels(model_type="gcn", dataset="cora", phase="test").set(test_acc)
    print(f"Test Accuracy: {test_acc:.4f}")


def _accuracy(model: GCN, data, mask) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
    return float((pred[mask] == data.y[mask]).sum()) / float(mask.sum())


if __name__ == "__main__":
    train()
