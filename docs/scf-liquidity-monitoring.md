# Deep-Dive: How AstroML Solves Liquidity Monitoring for the Stellar Community Fund (SCF)

The Stellar Community Fund (SCF) distributes millions of XLM to support developers and startups building on the Stellar network. A critical post-funding challenge is **Liquidity Monitoring**. When new projects launch features—particularly tokens and Automated Market Maker (AMM) pools—the SCF needs to ensure that:
1. **Liquidity is stable and healthy** (no sudden rug pulls).
2. **Trading volume is organic** (no wash trading or artificial volume inflation).
3. **Malicious actors are identified quickly** to protect users and the ecosystem.

**AstroML**, a dynamic graph machine learning framework designed specifically for the Stellar network, provides an end-to-end pipeline to solve these challenges. Instead of relying on static heuristics, AstroML models AMM pools and token flows as a **dynamic, time-evolving multi-asset graph**, enabling advanced Graph Neural Networks (GNNs) to detect deep structural anomalies.

This document explores how AstroML's architecture solves SCF liquidity monitoring.

---

## 1. Data Ingestion: The Foundation of Monitoring

AstroML's **Enhanced Stellar Ingestion Service** provides the robust data pipeline required for real-time liquidity monitoring.

### 1.1 Multi-Horizon Streaming
AstroML can stream operations and effects across multiple Horizon instances simultaneously. To monitor liquidity, the ingestion service captures:
- **`liquidity_pool_deposit` / `liquidity_pool_withdraw`**: Tracks absolute liquidity entering and leaving pools.
- **`path_payment_strict_receive` / `path_payment_strict_send` (Swaps)**: Tracks token volume and price impact across pools.
- **`manage_buy_offer` / `manage_sell_offer`**: Tracks orderbook depth.

### 1.2 Resilient State Management
Liquidity monitoring cannot afford downtime. AstroML's ingestion features adaptive backoff, connection health monitoring, and state persistence (cursors), ensuring that no crucial AMM state changes are missed during network congestion.

---

## 2. Dynamic Graph Construction: Modeling AMM Pools

Traditional analytics view liquidity pools in isolation (TVL, 24h volume). AstroML transforms this data into a **Dynamic Transaction Graph**.

### 2.1 Graph Structure
- **Nodes**: Accounts (Users, SCF Projects) and AMM Pools.
- **Edges**: Directed interactions (Deposits, Withdrawals, Swaps).
- **Edge Types**: Assets (XLM, USDC, Project Tokens).
- **Temporal Dimension**: Rolling time windows (e.g., 1hr, 24hr snapshots) track how the graph structure evolves.

### 2.2 Feature Engineering for Liquidity
AstroML computes node-level and graph-level features critical for monitoring:
- **Concentration Metrics**: What percentage of an AMM pool is owned by the top 3 nodes?
- **Velocity**: How frequently are LP shares being created and destroyed?
- **Cyclic Flow Ratios**: What portion of the volume is flowing in closed loops (a strong indicator of wash trading)?

---

## 3. Machine Learning Models: Detecting Liquidity Anomalies

AstroML utilizes state-of-the-art ML architectures to monitor the health of SCF projects.

### 3.1 Unsupervised Anomaly Detection (Deep SVDD)
Because true "rug pulls" or zero-day exploits are rare and unlabeled, AstroML employs **Deep Support Vector Data Description (Deep SVDD)** and **Isolation Forests**.
- **How it works for SCF:** The model learns a compact representation of "normal" liquidity provision behavior. If a project teams' wallets or connected smart contracts begin exhibiting structural behaviors that deviate from this norm (e.g., a highly coordinated, simultaneous withdrawal of LP tokens involving multiple obfuscated accounts), the model flags it as an anomaly.

### 3.2 Graph Convolutional Networks (GCNs) for Sybil Detection
Wash traders often create hundreds of Sybil accounts to trade back and forth, inflating volume to meet SCF traction metrics.
- **How it works for SCF:** GCNs analyze the graph's topology. Even if an attacker creates 1,000 accounts, those accounts will form a dense, anomalous clique in the transaction graph. The GCN identifies these structural subgraphs, flagging the inflated volume.

### 3.3 Privacy-Preserving Detection
If SCF needs to audit proprietary institutional market makers without revealing exact trade sizes or strategies, AstroML supports **Privacy-Preserving Anomaly Detection**. By using differential privacy or Localized Sensitive Hashing (LSH), AstroML can compute anomaly scores on liquidity flows while keeping exact counterparty interactions confidential.

---

## 4. On-Chain Action: The Soroban Fraud Registry

Detection is only the first half of the solution; action is the second. AstroML integrates with a **Soroban Fraud Registry Contract**.

### 4.1 Automated On-Chain Reporting
When AstroML's off-chain inference engine flags an account or liquidity pool for severe manipulation with high confidence, it can trigger a report to the Soroban smart contract.
- The AstroML backend acts as a highly reputable **Validator**.
- It invokes `report_fraud`, submitting the suspicious `account_id`, an anomaly score (`confidence`), and a hash of the GNN evidence subgraph (`evidence_hash`).

### 4.2 Ecosystem Protection
Once consensus is reached on the Soroban registry:
- **SCF Grant Disbursements can be paused automatically** via escrow contracts that check `is_fraudulent(account_id)`.
- **DEX UI/Wallets** can query the Soroban contract and warn users before they swap into a manipulated liquidity pool.

---

## 5. Summary of the AstroML Pipeline for SCF

1. **Ingest**: Stream Stellar operations targeting specific SCF project tokens and AMM pools.
2. **Build**: Construct rolling time-window graphs of all liquidity providers and swappers.
3. **Analyze**: Run GCN and Deep SVDD models to identify wash-trading cliques or impending rug-pull structures.
4. **Alert & Act**: Flag anomalies to the SCF committee dashboard, and push high-confidence fraud reports directly to the Soroban Fraud Registry to protect the broader Stellar ecosystem.

By treating liquidity not just as a financial metric but as a dynamic structural graph, AstroML provides the Stellar Community Fund with an unprecedented, deep-learning-powered shield against manipulation.
