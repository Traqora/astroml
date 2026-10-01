from astroml.utils.exceptions import AstroMLError
"""GraphSAGE model for node classification and transaction analysis.

Implements inductive representation learning and node classification on
transaction networks using GraphSAGE convolutions. Supports multiple
aggregation methods, multi-layer architectures, embedding extraction,
and comprehensive training/evaluation routines.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.nn import SAGEConv as PyGSAGEConv

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    PyGSAGEConv = None

from astroml.features.gnn.sage import SAGEConv as InternalSAGEConv

logger = logging.getLogger(__name__)


def _get_sage_conv(
    in_channels: int,
    out_channels: int,
    aggregator: str = "mean",
    bias: bool = True,
) -> nn.Module:
    """Instantiate a SAGEConv layer, preferring PyG if available."""
    if _HAS_PYG and PyGSAGEConv is not None:
        try:
            return PyGSAGEConv(
                in_channels=in_channels,
                out_channels=out_channels,
                aggr=aggregator,
                bias=bias,
            )
        except AstroMLError:
            pass
    return InternalSAGEConv(
        in_dim=in_channels,
        out_dim=out_channels,
        aggregator=aggregator,
        bias=bias,
    )


class GraphSAGENodeClassifier(nn.Module):
    """Multi-layer GraphSAGE neural network for node classification.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features.
    hidden_channels : int | Sequence[int]
        Hidden layer dimensionality. If an int, repeated for each hidden layer.
    out_channels : int
        Number of output classes (e.g. 2 for binary fraud vs legitimate).
    num_layers : int
        Total number of GraphSAGE layers (must be >= 1). Default is 2.
    dropout : float
        Dropout probability applied between layers. Default is 0.5.
    aggregator : str
        Aggregation strategy ('mean', 'gcn', 'max', etc.). Default is 'mean'.
    activation : str
        Activation function between layers ('relu', 'elu', 'leaky_relu', 'tanh').
    use_batch_norm : bool
        Whether to apply batch normalization between convolution layers.
    normalize_embeddings : bool
        Whether to L2-normalize node embeddings.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | Sequence[int],
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
        aggregator: str = "mean",
        activation: str = "relu",
        use_batch_norm: bool = False,
        normalize_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.aggregator = aggregator.lower()
        self.normalize_embeddings = normalize_embeddings
        self.use_batch_norm = use_batch_norm

        if isinstance(hidden_channels, int):
            hidden_dims = [hidden_channels] * (num_layers - 1)
        else:
            hidden_dims = list(hidden_channels)
            if len(hidden_dims) != num_layers - 1:
                raise ValueError(
                    f"Expected {num_layers - 1} hidden dimensions, got {len(hidden_dims)}"
                )

        layer_dims = [in_channels] + hidden_dims + [out_channels]
        self.layer_dims = layer_dims

        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None

        for i in range(num_layers):
            conv = _get_sage_conv(
                in_channels=layer_dims[i],
                out_channels=layer_dims[i + 1],
                aggregator=self.aggregator,
                bias=True,
            )
            self.convs.append(conv)
            if use_batch_norm and i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(layer_dims[i + 1]))

        # Activation lookup
        act_map = {
            "relu": F.relu,
            "elu": F.elu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
        }
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = act_map.get(
            activation.lower(), F.relu
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute logits for all nodes in the graph.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor of shape [N, in_channels].
        edge_index : torch.Tensor
            Graph edge index of shape [2, E].

        Returns
        -------
        torch.Tensor
            Logits tensor of shape [N, out_channels].
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:
                if self.batch_norms is not None:
                    x = self.batch_norms[i](x)
                x = self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                if self.normalize_embeddings:
                    x = F.normalize(x, p=2, dim=-1)

        return x

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract node embeddings up to a specified layer.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor [N, in_channels].
        edge_index : torch.Tensor
            Graph edge index [2, E].
        layer : int
            Layer index to extract embeddings from (-1 for penultimate or last representation).

        Returns
        -------
        torch.Tensor
            Node embedding representations [N, embedding_dim].
        """
        self.eval()
        target_layer = layer if layer >= 0 else (self.num_layers + layer)
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i == target_layer:
                    if self.normalize_embeddings:
                        x = F.normalize(x, p=2, dim=-1)
                    return x
                if i < self.num_layers - 1:
                    if self.batch_norms is not None:
                        x = self.batch_norms[i](x)
                    x = self.act_fn(x)
                    if self.normalize_embeddings:
                        x = F.normalize(x, p=2, dim=-1)
        return x

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class probabilities via softmax.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, in_channels].
        edge_index : torch.Tensor
            Graph edge index [2, E].

        Returns
        -------
        torch.Tensor
            Probabilities [N, out_channels].
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index)
            if self.out_channels == 1:
                return torch.sigmoid(logits)
            return F.softmax(logits, dim=-1)

    def predict(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Predict class labels.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, in_channels].
        edge_index : torch.Tensor
            Graph edge index [2, E].

        Returns
        -------
        torch.Tensor
            Predicted class indices [N].
        """
        probs = self.predict_proba(x, edge_index)
        if self.out_channels == 1:
            return (probs >= 0.5).long().squeeze(-1)
        return probs.argmax(dim=-1)


def train_graphsage_classifier(
    model: GraphSAGENodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    class_weights: torch.Tensor | None = None,
    early_stopping_patience: int = 20,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train GraphSAGE node classification model.

    Parameters
    ----------
    model : GraphSAGENodeClassifier
        Model to train.
    x : torch.Tensor
        Node features [N, in_channels].
    edge_index : torch.Tensor
        Edge index [2, E].
    labels : torch.Tensor
        Node target labels [N].
    train_mask : torch.Tensor
        Boolean mask of nodes for training [N].
    val_mask : torch.Tensor | None
        Boolean mask of nodes for validation [N].
    epochs : int
        Maximum training epochs.
    lr : float
        Learning rate for Adam optimizer.
    weight_decay : float
        Weight decay (L2 regularization).
    class_weights : torch.Tensor | None
        Weights per class for imbalanced loss computation.
    early_stopping_patience : int
        Patience epochs before early stopping.
    device : str
        Execution device ('cpu' or 'cuda').

    Returns
    -------
    dict[str, Any]
        Dictionary with training metrics, best validation score, and history.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)
    x = x.to(dev)
    edge_index = edge_index.to(dev)
    labels = labels.to(dev)
    train_mask = train_mask.to(dev)
    if val_mask is not None:
        val_mask = val_mask.to(dev)

    weights = class_weights.to(dev) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_weights: dict[str, Any] | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(x, edge_index)
        loss = criterion(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()

        preds = out[train_mask].argmax(dim=-1)
        train_acc = (preds == labels[train_mask]).float().mean().item()
        history["train_loss"].append(loss.item())
        history["train_acc"].append(train_acc)

        if val_mask is not None and val_mask.sum() > 0:
            model.eval()
            with torch.no_grad():
                val_out = model(x, edge_index)
                v_loss = criterion(val_out[val_mask], labels[val_mask]).item()
                v_preds = val_out[val_mask].argmax(dim=-1)
                v_acc = (v_preds == labels[val_mask]).float().mean().item()
                history["val_loss"].append(v_loss)
                history["val_acc"].append(v_acc)

                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    patience_counter = 0
                    best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

    if best_weights is not None:
        model.load_state_dict({k: v.to(dev) for k, v in best_weights.items()})

    return {
        "epochs_trained": len(history["train_loss"]),
        "final_train_loss": history["train_loss"][-1] if history["train_loss"] else 0.0,
        "final_train_acc": history["train_acc"][-1] if history["train_acc"] else 0.0,
        "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
        "final_val_acc": history["val_acc"][-1] if history["val_acc"] else None,
        "history": history,
    }


def evaluate_graphsage(
    model: GraphSAGENodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate GraphSAGE classifier on masked nodes.

    Parameters
    ----------
    model : GraphSAGENodeClassifier
    x : torch.Tensor
    edge_index : torch.Tensor
    labels : torch.Tensor
    mask : torch.Tensor
    device : str

    Returns
    -------
    dict[str, float]
        Metrics dictionary containing accuracy, loss, and precision/recall/f1 if binary.
    """
    dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    model = model.to(dev)
    model.eval()

    x = x.to(dev)
    edge_index = edge_index.to(dev)
    labels = labels.to(dev)
    mask = mask.to(dev)

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        out = model(x, edge_index)
        loss = criterion(out[mask], labels[mask]).item()
        preds = out[mask].argmax(dim=-1)
        targets = labels[mask]
        acc = (preds == targets).float().mean().item()

    y_true = targets.cpu().numpy()
    y_pred = preds.cpu().numpy()

    metrics = {
        "loss": float(loss),
        "accuracy": float(acc),
    }

    # Binary / Multiclass additional metrics
    if len(np.unique(y_true)) <= 2:
        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics.update(
            {
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "true_positives": float(tp),
                "false_positives": float(fp),
                "true_negatives": float(tn),
                "false_negatives": float(fn),
            }
        )

    return metrics
