"""Test fixture generation for LLM-generated tests.

Generates pytest fixtures, mock objects, and realistic
test data for various domains and use cases.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, Callable


@dataclass
class FixtureConfig:
    """Configuration for fixture generation."""
    domain: str = "general"
    include_mocks: bool = True
    realistic_data: bool = True
    num_samples: int = 5


class FixtureGenerator:
    """Generates test fixtures for various domains."""

    def __init__(self, config: Optional[FixtureConfig] = None):
        self.config = config or FixtureConfig()

    def generate_pytest_fixture(
        self,
        fixture_name: str,
        return_type: str = "dict",
    ) -> str:
        """Generate a pytest fixture function."""
        data = self._generate_sample_data(return_type)
        return (
            f"@pytest.fixture\n"
            f"def {fixture_name}() -> {return_type}:\n"
            f"    return {repr(data)}\n"
        )

    def generate_sqlalchemy_fixture(
        self,
        model_name: str,
        fields: Dict[str, str],
    ) -> str:
        """Generate a SQLAlchemy model fixture."""
        factory_name = f"create_{model_name.lower()}"
        params = ", ".join(f"{name}: {ftype}" for name, ftype in fields.items())

        return (
            f"@pytest.fixture\n"
            f"def {factory_name}(db_session) -> Callable[..., {model_name}]:\n"
            f"    def _factory({params}) -> {model_name}:\n"
            f"        instance = {model_name}(\n"
            + "\n".join(f"            {name}={name}," for name in fields) + "\n"
            f"        )\n"
            f"        db_session.add(instance)\n"
            f"        db_session.commit()\n"
            f"        return instance\n"
            f"    return _factory\n"
        )

    def generate_mock_fixture(self, class_name: str) -> str:
        """Generate a mock fixture for a class."""
        return (
            f"@pytest.fixture\n"
            f"def mock_{class_name.lower()}():\n"
            f"    mock = MagicMock(spec={class_name})\n"
            f"    return mock\n"
        )

    def _generate_sample_data(self, return_type: str) -> Any:
        if return_type == "dict":
            return {
                "id": str(uuid.uuid4()),
                "name": "test_name",
                "created_at": datetime.utcnow().isoformat(),
                "active": True,
            }
        elif return_type == "list":
            return [
                {"id": str(uuid.uuid4()), "value": i}
                for i in range(self.config.num_samples)
            ]
        elif return_type == "str":
            return "test_string"
        elif return_type in ("int", "float"):
            return 42
        elif return_type == "bool":
            return True
        elif return_type == "pd.DataFrame":
            import pandas as pd
            return pd.DataFrame({
                "id": [str(uuid.uuid4()) for _ in range(5)],
                "value": list(range(5)),
                "timestamp": [
                    datetime.utcnow() - timedelta(hours=i)
                    for i in range(5)
                ],
            }).to_dict()
        return None
