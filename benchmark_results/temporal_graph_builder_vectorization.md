# TemporalGraphBuilder edge construction benchmark

Issue: #769 Vectorize edge construction with NumPy/Polars.

Command used locally:

```bash
/root/.pyenv/versions/3.11.15/bin/python - <<'PY'
import random, time, tracemalloc
import torch
from astroml.utils.temporal import TemporalGraphBuilder

random.seed(7)
txs = [
    {
        "source_account": f"acct_{random.randrange(20_000)}",
        "target_account": f"acct_{random.randrange(20_000)}",
        "timestamp": float(i),
        "amount": float(random.randrange(1, 10_000)) / 100,
        "operation_type": random.choice(["payment", "path_payment", "create_account"]),
    }
    for i in range(50_000)
]

builder = TemporalGraphBuilder()
builder.processor.compute_temporal_features = lambda timestamps, features, window_size=10: features

for _ in range(3):
    builder.build_temporal_graph(txs)
tracemalloc.start()
start = time.perf_counter()
for _ in range(5):
    graph = builder.build_temporal_graph(txs)
elapsed = (time.perf_counter() - start) / 5
current, peak = tracemalloc.get_traced_memory()
print(f"elapsed_s={elapsed:.4f} peak_mb={peak / 1024 / 1024:.1f} edges={graph['edge_index'].size(1)} nodes={graph['num_nodes']}")
PY
```

| Implementation | Mean build time | Peak traced memory | Workload |
| --- | ---: | ---: | --- |
| Previous Python append/update loops | 1.2334 s | 39.5 MiB | 50k synthetic transactions / ~20k accounts |
| Vectorized Polars + NumPy tensors | 0.5887 s | 37.1 MiB | Same workload |

Result: 2.1x faster edge construction and 6.1% lower traced peak memory on the synthetic benchmark above. The benchmark monkey-patches temporal feature extraction so it isolates edge-list, edge-feature, node-timestamp, and base-node-feature construction.
