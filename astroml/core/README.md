# Core Abstractions

This module provides abstract base classes (ABCs) and protocols that define contracts for key services in AstroML, enabling:

- **Dependency Injection**: Swap implementations without changing calling code
- **Testing**: Use mock implementations in unit tests
- **Extensibility**: Add new implementations by extending ABCs
- **Type Safety**: Clear interface contracts with type hints

## Abstract Base Classes

### Ingestor

Abstract base class for data ingestion services.

```python
from astroml.core import Ingestor, IngestionResult

class CustomIngestor(Ingestor):
    def ingest(self, start, end, **kwargs) -> IngestionResult:
        # Implementation
        return IngestionResult(...)

    def get_status(self) -> Dict[str, Any]:
        # Implementation
        return {...}
```

### FeatureComputer

Abstract base class for feature computation services.

```python
from astroml.core import FeatureComputer
import pandas as pd

class CustomFeatureComputer(FeatureComputer):
    def compute(self, data, **kwargs) -> pd.DataFrame:
        # Implementation
        return df

    def get_feature_schema(self) -> Dict[str, Any]:
        # Implementation
        return {...}

    def validate_input(self, data) -> bool:
        # Implementation
        return True
```

### GraphBuilder

Abstract base class for graph construction services.

```python
from astroml.core import GraphBuilder, Graph

class CustomGraphBuilder(GraphBuilder):
    def build_graph(self, transactions, **kwargs) -> Graph:
        # Implementation
        return Graph(...)

    def get_graph_statistics(self, graph) -> Dict[str, Any]:
        # Implementation
        return {...}
```

## Mock Implementations

Mock implementations are provided for testing:

```python
from astroml.core.mocks import MockIngestor, MockFeatureComputer, MockGraphBuilder

# Use in tests
ingestor = MockIngestor()
result = ingestor.ingest(start=1, end=10)

computer = MockFeatureComputer()
features = computer.compute(data)

builder = MockGraphBuilder()
graph = builder.build_graph(transactions)
```

## Dependency Injection

Use ABCs to enable dependency injection:

```python
from typing import Protocol
from astroml.core import Ingestor

class Application:
    def __init__(self, ingestor: Ingestor):
        self.ingestor = ingestor

# Inject implementation
app = Application(CustomIngestor())

# Or use mock for testing
test_app = Application(MockIngestor())
```

## Protocols

Additional protocols for optional capabilities:

- **Cacheable**: Services that support caching
- **Observable**: Services that support metrics collection

```python
from astroml.core import Cacheable, Observable

class CacheableFeatureComputer(FeatureComputer, Cacheable):
    def cache_key(self, *args, **kwargs) -> str:
        return f"feature:{args[0]}"

    def invalidate_cache(self, key=None) -> None:
        # Implementation
        pass
```

## Usage Patterns

### 1. Service Factory Pattern

```python
def create_ingestor(config: Dict) -> Ingestor:
    if config["type"] == "stellar":
        return StellarIngestor(config)
    elif config["type"] == "mock":
        return MockIngestor()
    else:
        raise ValueError(f"Unknown ingestor type: {config['type']}")
```

### 2. Configuration-Based Selection

```python
INGESTOR_IMPLEMENTATIONS = {
    "stellar": StellarIngestor,
    "mock": MockIngestor,
}

def get_ingestor(name: str) -> Ingestor:
    cls = INGESTOR_IMPLEMENTATIONS[name]
    return cls()
```

### 3. Testing with ABCs

```python
def test_feature_computation():
    # Use mock instead of real implementation
    computer = MockFeatureComputer()
    data = pd.DataFrame({"value": [1, 2, 3]})
    result = computer.compute(data)
    assert result is not None
```

## Extending ABCs

To add a new implementation:

1. Extend the ABC
2. Implement all abstract methods
3. Add optional protocol methods if needed
4. Register in factory if using dependency injection

```python
from astroml.core import Ingestor

class NewDataSourceIngestor(Ingestor):
    def ingest(self, start, end, **kwargs) -> IngestionResult:
        # Custom implementation
        pass

    def get_status(self) -> Dict[str, Any]:
        # Custom implementation
        pass
```

## Type Checking

ABCs enable static type checking:

```python
from astroml.core import Ingestor

def process_data(ingestor: Ingestor):
    result = ingestor.ingest(start=1, end=100)
    # Type checker knows result is IngestionResult
```
