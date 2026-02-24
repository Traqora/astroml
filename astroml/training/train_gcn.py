import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from astroml.models.gcn import GCN


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = Planetoid(root="data", name="Cora", transform=NormalizeFeatures())
    data = dataset[0].to(device)

    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dims=[64],
        output_dim=dataset.num_classes,
        dropout=0.5,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    test(model, data)


def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f"Test Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()