"""Great Expectations integration for automated data validation.

Resolves #644.

Great Expectations itself is an optional dependency.  Suites built here use the
GE expectation vocabulary and convert to native GE objects via
:meth:`ExpectationSuite.to_great_expectations` when the package is installed;
without it, the in-repo engine in
:mod:`astroml.validation.great_expectations.validator` executes the same suites
so validation still runs in the slim CI image.

Example::

    suite = SuiteBuilder.from_dataset("transactions", dataframe)
    result = DataValidator(suite).validate(dataframe)
    ValidationStore("validation_results").save(result)
    DataDocsBuilder().build(suites=[suite], results=[result])
"""

from __future__ import annotations

from astroml.validation.great_expectations.data_docs import DataDocsBuilder, DataDocsPage
from astroml.validation.great_expectations.suite_builder import (
    Expectation,
    ExpectationSuite,
    ExpectationType,
    SuiteBuilder,
    great_expectations_available,
)
from astroml.validation.great_expectations.validator import (
    DataValidationError,
    DataValidator,
    ExpectationResult,
    ValidationResult,
    ValidationStore,
)

__all__ = [
    "DataDocsBuilder",
    "DataDocsPage",
    "DataValidationError",
    "DataValidator",
    "Expectation",
    "ExpectationResult",
    "ExpectationSuite",
    "ExpectationType",
    "SuiteBuilder",
    "ValidationResult",
    "ValidationStore",
    "great_expectations_available",
]
