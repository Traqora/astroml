from typing import Any, Dict, List, Optional, Union, Callable
"""Pipeline testing framework for data quality, schema validation, and integrity checks.

Issue #638: Comprehensive testing framework for data pipelines.
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    "DataTestSuite",
    "DataTestResult",
    "DataAssertion",
    "SchemaValidator",
    "SchemaValidationResult",
    "IntegrityChecker",
    "IntegrityReport",
    "PipelineFixture",
    "PipelineTestRunner",
    "DataDiffReport",
    "RegressionTest",
]

_LAZY: dict[str, tuple[str, str]] = {
    "DataTestSuite": ("astroml.pipeline.testing.data_tests", "DataTestSuite"),
    "DataTestResult": ("astroml.pipeline.testing.data_tests", "DataTestResult"),
    "DataAssertion": ("astroml.pipeline.testing.data_tests", "DataAssertion"),
    "SchemaValidator": ("astroml.pipeline.testing.schema_validator", "SchemaValidator"),
    "SchemaValidationResult": ("astroml.pipeline.testing.schema_validator", "SchemaValidationResult"),
    "IntegrityChecker": ("astroml.pipeline.testing.integrity", "IntegrityChecker"),
    "IntegrityReport": ("astroml.pipeline.testing.integrity", "IntegrityReport"),
    "PipelineFixture": ("astroml.pipeline.testing.fixtures", "PipelineFixture"),
    "PipelineTestRunner": ("astroml.pipeline.testing.fixtures", "PipelineTestRunner"),
    "DataDiffReport": ("astroml.pipeline.testing.data_tests", "DataDiffReport"),
    "RegressionTest": ("astroml.pipeline.testing.data_tests", "RegressionTest"),
}


def __getattr__(name -> Any: str):
    if name in _LAZY:
        module_path, attr = _LAZY[name]
        module = import_module(module_path)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")