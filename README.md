# AstroML

## Dynamic Graph Machine Learning Framework for the Stellar Network

**AstroML** is a research-driven Python framework for building **dynamic graph machine learning models** on the Stellar Development Foundation Stellar blockchain.

It treats blockchain data as a **multi-asset, time-evolving graph**, enabling advanced ML research on transaction networks such as fraud detection, anomaly detection, and behavioral modeling.

---

## ✨ Features

AstroML provides end-to-end tooling for:

* Ledger ingestion and normalization
* Dynamic transaction graph construction
* Feature engineering for blockchain accounts
* Graph Neural Networks (GNNs)
* Self-supervised node embeddings
* Anomaly detection
* Temporal modeling
* Reproducible ML experimentation

---

## 🧠 Core Idea

Blockchain networks are naturally **graph-structured systems**:

| Blockchain Concept | Graph Representation |
| ------------------ | -------------------- |
| Accounts           | Nodes                |
| Transactions       | Directed edges       |
| Assets             | Edge types           |
| Time               | Dynamic dimension    |

Most analytics tools rely on static heuristics or SQL queries.

**AstroML instead enables:**

* Dynamic graph learning
* Temporal GNNs
* Representation learning
* Research-grade experimentation

---

## 🎯 Target Users

AstroML is designed for:

* ML researchers
* Graph ML engineers
* Fraud detection teams
* Blockchain data scientists

---

## 🏗 Architecture Overview

```
Ledger → Ingestion → Normalization → Graph Builder → Features → GNN/ML Models → Experiments
```


## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Traqora/astroml.git
cd astroml
```

### 2. Create environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure database

Create a PostgreSQL database and update:

```
config/database.yaml
```

---

## 📥 Data Ingestion

Backfill ledgers:

```bash
python -m astroml.ingestion.backfill \
  --start-ledger 1000000 \
  --end-ledger 1100000
```

---

## 🕸 Build Graph Snapshot

Create a rolling time window graph:

```bash
python -m astroml.graph.build_snapshot --window 30d
```

---

## 🤖 Train Baseline GCN

```bash
python -m astroml.training.train_gcn
```

---

## 📊 Example Use Cases

* Fraud / scam detection
* Account clustering
* Transaction risk scoring
* Temporal behavior modeling
* Self-supervised embeddings
* Network anomaly detection

---

## 🔬 Research Goals

AstroML emphasizes:

* Reproducibility
* Modular experimentation
* Scalable ingestion
* Temporal graph learning
* Production-ready ML pipelines

---

## 🛠 Tech Stack

* Python
* PyTorch / PyTorch Geometric
* PostgreSQL
* NetworkX / graph tooling

---

## 📌 Roadmap

* [ ] Real-time streaming ingestion
* [ ] Temporal GNN models
* [ ] Contrastive learning pipelines
* [ ] Feature store
* [ ] Model benchmarking suite
* [ ] Docker deployment

---

## 🤝 Contributing

Contributions are welcome!

```bash
fork → branch → commit → PR
```

Please open issues for bugs or feature requests.

---

## 📜 License

MIT License


