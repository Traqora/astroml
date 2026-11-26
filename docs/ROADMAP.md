# Roadmap

This page is the single source of truth for features that are **not yet
supported** in AstroML. Claims about unsupported or planned capabilities that
used to live inline in feature docs have been moved here so the onboarding path
in [ONBOARDING.md](ONBOARDING.md) and the quick start in
[QUICKSTART_GUIDE.md](../QUICKSTART_GUIDE.md) describe only what works on a
clean machine today.

The split this page enforces is automated:
`scripts/check_roadmap_consolidation.py` fails if a feature doc still carries
an inline `## Future Enhancements` or `### Planned Features` section, or if the
README roadmap lists a capability that is already shipped as a to-do. Update
this page in the same change that adds or removes a planned feature.

## Status legend

- **Not yet supported** — no shipped implementation. Do not present this as a
  current feature in onboarding, quick start, or the README feature list.
- **Partial** — code exists behind a flag or profile but it is not part of the
  documented v1 onboarding path.
- **Supported** — shipped and covered by the v1 workflow.

## Ingestion and streaming

| Capability | Status | Notes |
| --- | --- | --- |
| Real-time streaming ingestion | Not yet supported | Ingestion is batch/incremental backfill (see [ONBOARDING.md](ONBOARDING.md)). Continuous streaming is planned. |
| Real-time validation of streaming data | Not yet supported | Tracked with the streaming work above; validation today runs on batches. |

## Modeling

| Capability | Status | Notes |
| --- | --- | --- |
| Temporal GNN models | Not yet supported | Static GNN baselines (GCN, GraphSAGE, GAT) ship today; temporal GNN architectures are planned. |
| Contrastive learning pipelines | Not yet supported | No shipped contrastive objective yet. |

## Data quality validation

The following are planned improvements to the data quality validation
framework (see [DATA_QUALITY_VALIDATION.md](DATA_QUALITY_VALIDATION.md) for what
ships today):

- **Machine Learning Validation** — ML-based anomaly detection.
- **Custom Rule Engine** — user-defined validation rules.
- **Performance Optimization** — parallel processing for large datasets.
- **Enhanced Reporting** — more detailed analytics and visualization.
- **Pipeline monitoring integration** — tighter coupling with data pipeline
  monitoring tools.

## Soroban fraud registry contract

The following are planned contract features (see
[FRAUD_REGISTRY_CONTRACT.md](FRAUD_REGISTRY_CONTRACT.md) for the current
contract surface):

- **Event Logging** — emit events for all state changes to enable off-chain
  monitoring.
- **Time-Based Expiry** — automatic report expiry, reputation decay, and
  appeal time limits.
- **Multi-Sig Admin** — require multiple admin signatures for distributed
  governance.
- **Staking Mechanism** — validator staking requirements, slashing, and
  economic incentives.

## Already supported (removed from the to-do list)

The README roadmap previously listed the following as unchecked to-dos. They
have shipped and are listed here only to keep the history honest — they are
**not** roadmap items:

- Feature store — shipped; see [FEATURE_STORE.md](FEATURE_STORE.md).
- Model benchmarking suite — shipped; see [benchmarking.md](benchmarking.md).
- Docker deployment — shipped; see [DOCKER.md](../DOCKER.md) and
  [docker-deployment.md](docker-deployment.md).

## Contributing a roadmap item

Open an issue labelled `feature request` and describe the capability and its
motivation. A maintainer will either accept it onto this page or close it with
a reason. Do not add a capability to the README feature list until it is
shipped and covered by the v1 workflow — that is what
`scripts/check_roadmap_consolidation.py` enforces.

## Related documentation

- [Onboarding](ONBOARDING.md) — the v1 clean-machine path.
- [Configuration reference](CONFIGURATION.md).
- [Which workflows need PostgreSQL](CI_DATABASE_REQUIREMENTS.md).
- [Troubleshooting](TROUBLESHOOTING.md).
