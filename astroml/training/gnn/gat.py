from astroml.utils.exceptions import AstroMLError
"""Graph Attention Network (GAT) model for node classification and transaction analysis.

Implements multi-head graph attention networks for relational transaction networks,
fraud detection, and explainable relationship mining with exportable attention weights.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from torch_geometric.nn import GATConv as PyGGATConv

    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    PyGGATConv = None

from astroml.features.gnn.attention import GATConv as InternalGATConv

logger = logging.getLogger(__name__)


def _get_gat_conv(
    in_channels: int,
    out_channels: int,
    heads: int = 4,
    concat: bool = True,
    dropout: float = 0.0,
    negative_slope: float = 0.2,
    bias: bool = True,
) -> nn.Module:
    """Instantiate a GATConv layer, preferring PyG if available."""
    if _HAS_PYG and PyGGATConv is not None:
        try:
            return PyGGATConv(
                in_channels=in_channels,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                negative_slope=negative_slope,
                dropout=dropout,
                bias=bias,
            )
        except AstroMLError:
            pass
    return InternalGATConv(
        in_dim=in_channels,
        out_dim=out_channels,
        heads=heads,
        concat=concat,
        dropout=dropout,
        negative_slope=negative_slope,
        bias=bias,
    )


class GATNodeClassifier(nn.Module):
    """Multi-layer Graph Attention Network for node classification.

    Parameters
    ----------
    in_channels : int
        Dimension of input node features.
    hidden_channels : int | Sequence[int]
        Hidden channels per attention head.
    out_channels : int
        Number of output classes.
    heads : int
        Number of attention heads in hidden layers. Default is 4.
    out_heads : int
        Number of attention heads in final layer. Default is 1.
    num_layers : int
        Total number of GAT layers (must be >= 1). Default is 2.
    dropout : float
        Dropout probability for features and attention weights. Default is 0.5.
    negative_slope : float
        LeakyReLU negative slope for attention mechanism. Default is 0.2.
    activation : str
        Activation function between layers ('elu', 'relu', 'leaky_relu').
    residual : bool
        Whether to add skip/residual connections when layer input/output dimensions match.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int | Sequence[int],
        out_channels: int,
        heads: int = 4,
        out_heads: int = 1,
        num_layers: int = 2,
        dropout: float = 0.5,
        negative_slope: float = 0.2,
        activation: str = "elu",
        residual: bool = False,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be at least 1, got {num_layers}")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.out_heads = out_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.negative_slope = negative_slope
        self.residual = residual

        if isinstance(hidden_channels, int):
            hidden_dims = [hidden_channels] * (num_layers - 1)
        else:
            hidden_dims = list(hidden_channels)
            if len(hidden_dims) != num_layers - 1:
                raise ValueError(
                    f"Expected {num_layers - 1} hidden dimensions, got {len(hidden_dims)}"
                )

        self.convs = nn.ModuleList()
        self.residual_linears = nn.ModuleList() if residual else None

        current_dim = in_channels
        for i in range(num_layers - 1):
            h_dim = hidden_dims[i]
            conv = _get_gat_conv(
                in_channels=current_dim,
                out_channels=h_dim,
                heads=heads,
                concat=True,
                dropout=dropout,
                negative_slope=negative_slope,
                bias=True,
            )
            self.convs.append(conv)
            if residual and self.residual_linears is not None:
                if current_dim != h_dim * heads:
                    self.residual_linears.append(nn.Linear(current_dim, h_dim * heads, bias=False))
                else:
                    self.residual_linears.append(nn.Identity())
            current_dim = h_dim * heads

        # Final classification layer
        final_conv = _get_gat_conv(
            in_channels=current_dim,
            out_channels=out_channels,
            heads=out_heads,
            concat=False,
            dropout=dropout,
            negative_slope=negative_slope,
            bias=True,
        )
        self.convs.append(final_conv)

        # Activation lookup
        act_map = {
            "elu": F.elu,
            "relu": F.relu,
            "leaky_relu": F.leaky_relu,
            "tanh": torch.tanh,
        }
        self.act_fn: Callable[[torch.Tensor], torch.Tensor] = act_map.get(activation.lower(), F.elu)

        self.last_attention_weights: tuple[torch.Tensor, torch.Tensor] | None = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward computation for node classification.

        Parameters
        ----------
        x : torch.Tensor
            Node features [N, in_channels].
        edge_index : torch.Tensor
            Edge index [2, E].
        return_attention_weights : bool
            Whether to return attention weights of the first layer.

        Returns
        -------
        torch.Tensor | tuple
            Logits [N, out_channels] or (logits, (edge_index, alpha)).
        """
        first_attention = None

        for i, conv in enumerate(self.convs):
            prev_x = x
            if hasattr(conv, "export_attention") or hasattr(conv, "forward"):
                # Handle return_attention if supported by conv
                try:
                    if return_attention_weights and i == 0:
                        res = conv(x, edge_index, return_attention=True)
                        if isinstance(res, tuple):
                            x, alpha = res
                            first_attention = (edge_index, alpha)
                            self.last_attention_weights = first_attention
                        else:
                            x = res
                    else:
                        x = conv(x, edge_index)
                except TypeError:
                    x = conv(x, edge_index)
            else:
                x = conv(x, edge_index)

            if i < self.num_layers - 1:
                if self.residual and self.residual_linears is not None:
                    res_x = self.residual_linears[i](prev_x)
                    x = x + res_x
                x = self.act_fn(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if return_attention_weights:
            if first_attention is None and hasattr(self.convs[0], "export_attention"):
                first_attention = self.convs[0].export_attention()
            return x, first_attention
        return x

    def get_attention_weights(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Export edge attention weights for inspection and relationship mining."""
        self.eval()
        with torch.no_grad():
            _, attn = self.forward(x, edge_index, return_attention_weights=True)
            return attn

    def get_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Extract node embeddings from the penultimate layer."""
        self.eval()
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                if i == self.num_layers - 1:
                    return x
                prev_x = x
                x = conv(x, edge_index)
                if self.residual and self.residual_linears is not None:
                    x = x + self.residual_linears[i](prev_x)
                x = self.act_fn(x)
        return x

    def predict_proba(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Compute class probabilities."""
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
        """Predict class labels."""
        probs = self.predict_proba(x, edge_index)
        if self.out_channels == 1:
            return (probs >= 0.5).long().squeeze(-1)
        return probs.argmax(dim=-1)


def train_gat_classifier(
    model: GATNodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    val_mask: torch.Tensor | None = None,
    epochs: int = 100,
    lr: float = 0.005,
    weight_decay: float = 5e-4,
    class_weights: torch.Tensor | None = None,
    early_stopping_patience: int = 20,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train GAT node classification model.

    Parameters
    ----------
    model : GATNodeClassifier
    x : torch.Tensor
    edge_index : torch.Tensor
    labels : torch.Tensor
    train_mask : torch.Tensor
    val_mask : torch.Tensor | None
    epochs : int
    lr : float
    weight_decay : float
    class_weights : torch.Tensor | None
    early_stopping_patience : int
    device : str

    Returns
    -------
    dict[str, Any]
        Training summary with loss and accuracy curves.
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


def evaluate_gat(
    model: GATNodeClassifier,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    device: str = "cpu",
) -> dict[str, float]:
    """Evaluate GAT classifier on masked nodes."""
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
