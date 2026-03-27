# Contributing to AstroML

Thank you for your interest in contributing to AstroML! This document provides guidelines and instructions for contributing code, documentation, and research to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Research to Production Workflow](#research-to-production-workflow)
- [Development Setup](#development-setup)
- [Code Standards](#code-standards)
- [Testing Requirements](#testing-requirements)
- [PR Process](#pr-process)
- [Documentation](#documentation)
- [Questions & Support](#questions--support)

---

## Code of Conduct

AstroML is committed to providing a welcoming and inclusive environment. All contributors are expected to:

- Be respectful and constructive in all interactions
- Welcome feedback and criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/astroml.git
cd astroml
git remote add upstream https://github.com/Traqora/astroml.git
```

### 2. Create a Feature Branch

```bash
# Sync with latest upstream
git fetch upstream
git checkout -b feature/your-feature-name upstream/main

# Or for bug fixes:
git checkout -b fix/bug-description upstream/main
```

### 3. Set Up Development Environment

See [Development Setup](#development-setup) section below.

---

## Research to Production Workflow

AstroML follows a clear data pipeline model that moves research from exploration to production. Understanding this workflow is essential for contributing effectively.

### The Data Pipeline

```
Ledger Data
    ↓
Ingestion & Normalization
    ↓
Graph Construction
    ↓
Feature Engineering
    ↓
Model Training & Evaluation
    ↓
Experimentation & Deployment
```

### Component Breakdown

| Stage | Module | Purpose | Examples |
|-------|--------|---------|----------|
| **Ingestion** | `astroml.ingestion` | Fetch ledgers from Stellar Horizon | `backfill`, `enhanced_stream` |
| **Normalization** | `astroml.ingestion` | Validate & deduplicate data | Duplicate removal, type conversion |
| **Graph Building** | `astroml.graph` | Construct transaction graphs | `build_snapshot`, windowing logic |
| **Features** | `astroml.features` | Extract node/edge features | Asset diversity, temporal decay, node importance |
| **Models** | `astroml.models` | GNN architectures & embeddings | GCN, GAT, GraphSAGE |
| **Training** | `astroml.training` | Model training pipelines | Config-driven experiments, checkpoints |

### Contributing to Each Stage

**When adding ingestion logic:**
- Ensure idempotency (re-runs are safe)
- Handle database constraints gracefully
- Test with small ledger ranges first
- Document config requirements in `config/database.yaml`

**When building graph features:**
- Test windowing logic thoroughly
- Ensure reproducibility (random seeds, checksums)
- Validate against edge cases (empty graphs, single nodes)
- Add unit tests before integration

**When creating models:**
- Use config files for hyperparameters (see `configs/`)
- Store checkpoints with metadata
- Log metrics consistently
- Provide examples in `examples/`

---

## Development Setup

### Prerequisites

- **Python 3.10+**
- **PostgreSQL 12+** (for ingestion tests; SQLite for unit tests)
- **Git**

### Installation

```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) CPU-only PyTorch
pip install -r requirements-cpu.txt

# 4. Configure database
cp config/database.yaml.example config/database.yaml
# Edit config/database.yaml with your PostgreSQL credentials

# 5. Install package in editable mode
pip install -e .

# 6. Run tests to verify setup
pytest tests/ -v
```

### Database Setup (for integration tests)

```bash
# Create a test database
createdb astroml_test

# Update config/database.yaml to point to test database
# Then run migrations:
alembic upgrade head
```

---

## Code Standards

### Python Style

AstroML follows **PEP 8** with these conventions:

- **Line length**: 88 characters (Black formatter)
- **Imports**: Organize as (stdlib, third-party, local)
- **Docstrings**: Use Google-style docstrings for all public functions/classes

#### Example:

```python
from datetime import datetime
from typing import Optional

import pandas as pd
from sqlalchemy import Column, String, Integer
from sqlalchemy.orm import declarative_base

from astroml.db.session import Base


def calculate_node_importance(
    graph: 'nx.DiGraph',
    measure: str = 'betweenness',
) -> dict:
    """Calculate node importance metrics for a transaction graph.
    
    Args:
        graph: NetworkX directed graph of transactions
        measure: One of 'betweenness', 'degree', 'closeness'
        
    Returns:
        Dictionary mapping node IDs to importance scores
        
    Raises:
        ValueError: If measure is not recognized
    """
    if measure not in ('betweenness', 'degree', 'closeness'):
        raise ValueError(f"Unknown measure: {measure}")
    
    # Implementation
    return {}
```

### Type Hints

- Use type hints for all function parameters and return types
- Import from `typing` module for complex types

```python
from typing import List, Dict, Optional, Tuple

def process_accounts(
    accounts: List[str],
    filters: Optional[Dict[str, int]] = None,
) -> Tuple[int, List[str]]:
    """Process a list of account IDs."""
    pass
```

### Naming Conventions

- **Functions/variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: Prefix with `_`

```python
class TransactionGraph:
    DEFAULT_WINDOW_SIZE = 30  # days
    
    def __init__(self):
        self._cache = {}
    
    def get_node_count(self) -> int:
        """Return number of nodes."""
        pass
```

### Comments & Documentation

- Write comments that explain **why**, not **what**
- Use docstrings for all public APIs
- Keep comments concise and up-to-date

```python
# Good: explains reasoning
# Use cached result if available to avoid re-querying Stellar Horizon
if node_id in self._cache:
    return self._cache[node_id]

# Avoid: obvious from code
# increment counter
count += 1
```

---

## Testing Requirements

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_schema.py -v

# Run with coverage
pytest tests/ --cov=astroml --cov-report=html

# Run async tests (marked with @pytest.mark.asyncio)
pytest tests/test_stream.py -v
```

### Writing Tests

**Test file naming**: `test_<module_name>.py`

```python
import pytest
from astroml.features import calculate_asset_diversity


class TestAssetDiversity:
    """Tests for asset diversity feature calculation."""
    
    def test_single_asset(self):
        """Single asset should have diversity = 1."""
        result = calculate_asset_diversity(['USD'])
        assert result == 1.0
    
    def test_empty_assets(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError):
            calculate_asset_diversity([])
    
    @pytest.mark.asyncio
    async def test_async_feature_extraction(self):
        """Test async feature pipeline."""
        result = await extract_features_async([...])
        assert len(result) > 0


@pytest.fixture
def sample_graph():
    """Fixture providing sample transaction graph."""
    import networkx as nx
    G = nx.DiGraph()
    G.add_edges_from([('A', 'B'), ('B', 'C')])
    return G
```

### Test Checklist

Before submitting a PR:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] New tests added for new functionality
- [ ] Edge cases covered (empty inputs, None values, etc.)
- [ ] Async functions tested with `@pytest.mark.asyncio`
- [ ] Integration tests verify database interactions
- [ ] No hardcoded test data paths (use fixtures)

### Testing Different Stages

| Stage | Test Type | Command |
|-------|-----------|---------|
| Ingestion | Unit + Integration | `pytest tests/test_*stream*.py` |
| Graph Building | Unit + Snapshot | `pytest tests/test_snapshot.py` |
| Features | Unit + Functional | `pytest tests/test_*features*.py` |
| Models | Unit + Training | `pytest tests/test_*.py -k model` |

---

## PR Process

### Before Opening a PR

1. **Sync with upstream:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run linting & tests locally:**
   ```bash
   # Check for obvious issues
   python -m py_compile astroml/**/*.py
   
   # Run full test suite
   pytest tests/ -v
   ```

3. **Ensure commits are clean:**
   - Meaningful commit messages (see [Commit Convention](#commit-convention))
   - Logical, separated changes
   - No secrets or credentials

### Commit Convention

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, `perf`

**Scope**: `ingestion`, `graph`, `features`, `models`, `training`, `db`

**Examples:**

```
feat(features): add temporal decay feature extractor

- Implements exponential decay based on transaction age
- Configured via decay_rate parameter
- Tested with synthetic graphs

Closes #123
```

```
fix(ingestion): handle duplicate transaction deduplication

Fixes idempotency issue when re-running backfill on same ledger range.

Fixes #456
```

### PR Template

When opening a PR, fill out:

```markdown
## Description
Brief description of what this PR does.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Related Issue
Closes #<issue_number>

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Tested against sample data

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-reviewed the code
- [ ] Updated documentation
- [ ] No new warnings generated
```

### Review Process

**Expectations:**

- Reviewers will provide feedback constructively
- Critical feedback focuses on the code, not the person
- Contributors should respond to all feedback (even if just acknowledging)
- Approval requires at least one maintainer sign-off

**What reviewers check:**

- ✅ Code correctness and logic
- ✅ Test coverage (especially for pipeline stages)
- ✅ Reproducibility (configs, seeds, checksums)
- ✅ Documentation completeness
- ✅ Alignment with "Research to Production" workflow
- ✅ Database integrity (for ingestion changes)

---

## Documentation

### Docstring Requirements

All public functions, classes, and modules must have docstrings:

```python
"""Module for extracting temporal features from transaction graphs.

This module implements exponential decay and recency weighting
for node features based on transaction timestamps.
"""

def calculate_temporal_decay(
    transactions: List[Transaction],
    decay_rate: float = 0.1,
) -> pd.DataFrame:
    """Calculate temporal decay weights for accounts.
    
    Uses exponential decay: weight = exp(-decay_rate * age_in_days)
    
    Args:
        transactions: List of Transaction objects (sorted by time)
        decay_rate: Decay coefficient (higher = faster decay)
        
    Returns:
        DataFrame with columns: [account_id, decay_weight, timestamp]
        
    Raises:
        ValueError: If decay_rate is negative or transactions list is empty
        
    Examples:
        >>> df = calculate_temporal_decay(transactions, decay_rate=0.1)
        >>> df.shape
        (1000, 3)
    """
```

### README Updates

When adding new features, update [README.md](README.md):

- Add to feature list if it's major functionality
- Update architecture diagram if pipeline changes
- Link to new example scripts or documentation

### Example Scripts

For new features, add an example in `examples/`:

```python
# examples/temporal_decay_example.py
"""Example: Extract temporal decay features."""

from astroml.features.temporal_decay import calculate_temporal_decay
from astroml.db.session import get_session

# Fetch transactions
session = get_session()
transactions = session.query(Transaction).all()

# Calculate temporal features
decay_df = calculate_temporal_decay(transactions, decay_rate=0.1)

print(f"Extracted temporal features for {len(decay_df)} accounts")
print(decay_df.head())
```

### Configuration Documentation

Document YAML config fields in docstrings:

```python
"""
Expected config (config/database.yaml):
    
    database:
      host: localhost
      port: 5432
      user: postgres
      password: ${DB_PASSWORD}  # From environment
      database: astroml
"""
```

---

## Questions & Support

- **Bug reports**: Open an issue on GitHub with reproducible example
- **Feature requests**: Use GitHub Discussions or open an issue with `[FEATURE]` tag
- **Questions**: Post in GitHub Discussions or tag with `[QUESTION]`
- **Security issues**: Email maintainers privately (do not open public issue)

### Getting Help

1. **Check existing issues/discussions** for similar questions
2. **Search the documentation** in `docs/` and README
3. **Review example scripts** in `examples/`
4. **Run the discovery checklist** from [copilot-instructions.md](.github/copilot-instructions.md)

---

## Additional Resources

- [README.md](README.md) - Project overview and quick start
- [docs/](docs/) - Full documentation
- [examples/](examples/) - Example scripts for common tasks
- [alembic/versions/](alembic/versions/) - Database migration history
- [configs/](configs/) - Example configuration files

---

## Thank You! 🙏

Your contributions make AstroML better for the entire research community. Whether you're fixing bugs, adding features, or improving documentation, every contribution matters.

**Happy coding!**
