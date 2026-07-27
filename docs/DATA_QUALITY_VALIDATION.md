# Data Quality Validation Framework

This document describes the comprehensive data quality validation framework added to AstroML, which provides extensive validation capabilities beyond the basic corruption detection.

## Overview

The data quality validation framework includes:

1. **Temporal Consistency Validation** - Timestamp ordering and future timestamp detection
2. **Referential Integrity Validation** - Account and asset format validation, ledger sequence checks
3. **Business Rules Validation** - Fee, amount, operation count, and balance validation
4. **Statistical Validation** - Outlier detection, timestamp gap analysis, duplicate pattern detection
5. **Comprehensive Validation** - Integrated validation pipeline with reporting

## Architecture

### Core Components

#### `DataQualityValidator`
The main orchestrator that combines all validation types into a comprehensive validation pipeline.

#### `TemporalValidator`
Validates temporal aspects of transaction data:
- Monotonic timestamp ordering within batches
- Future timestamp detection with configurable tolerance
- Timestamp format validation

#### `ReferentialIntegrityValidator`
Validates referential integrity and format compliance:
- Stellar account address format validation (G + 56 alphanumeric chars)
- Asset code format validation (1-12 alphanumeric chars)
- Ledger sequence positivity validation

#### `BusinessRulesValidator`
Validates domain-specific business rules:
- Non-negative fee validation
- Non-negative amount validation
- Operation count bounds (1-100 for Stellar)
- Balance format validation (no NaN/infinite values)

#### `StatisticalValidator`
Performs statistical data quality checks:
- Amount outlier detection using IQR method
- Timestamp gap analysis
- Duplicate pattern detection

### Data Structures

#### `ValidationResult`
Standard result structure for individual validation checks:
```python
@dataclass
class ValidationResult:
    is_valid: bool
    error_type: Optional[str] = None
    message: Optional[str] = None
    field: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
```

#### `DataQualityReport`
Comprehensive report for batch validation:
```python
@dataclass
class DataQualityReport:
    total_records: int = 0
    valid_records: int = 0
    validation_results: List[ValidationResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def quality_score(self) -> float:
        """Calculate data quality score as percentage of valid records."""
```

## Usage Examples

### Basic Validation

```python
from astroml.validation.data_quality import DataQualityValidator

validator = DataQualityValidator()
transactions = [...]  # Your transaction data

report = validator.validate_batch(transactions)
print(f"Quality Score: {report.quality_score:.1f}%")
print(f"Total Records: {report.total_records}")
print(f"Error Types: {report.error_types}")
```

### Individual Validation Types

```python
from astroml.validation.data_quality import (
    TemporalValidator,
    ReferentialIntegrityValidator,
    BusinessRulesValidator,
    StatisticalValidator
)

# Temporal validation
temporal_validator = TemporalValidator()
result = temporal_validator.validate_timestamp_ordering(transactions)

# Referential integrity
ref_validator = ReferentialIntegrityValidator()
account_result = ref_validator.validate_account_format("GABC...")

# Business rules
biz_validator = BusinessRulesValidator()
fee_result = biz_validator.validate_fee_non_negative(100)

# Statistical validation
stat_validator = StatisticalValidator()
outlier_result = stat_validator.detect_amount_outliers(amounts)
```

### Convenience Functions

```python
from astroml.validation.data_quality import (
    validate_data_quality,
    check_temporal_consistency,
    check_referential_integrity
)

# Comprehensive validation
report = validate_data_quality(transactions)

# Specific validation types
temporal_results = check_temporal_consistency(transactions)
referential_results = check_referential_integrity(transactions)
```

## Validation Rules

### Temporal Consistency

1. **Timestamp Ordering**: Timestamps within a batch should be monotonically increasing
2. **Future Timestamps**: No timestamps significantly in the future (configurable tolerance)
3. **Format Validation**: Timestamps must be valid ISO 8601 format

### Referential Integrity

1. **Account Format**: Stellar accounts must match `^G[A-Z0-9]{56}$` pattern
2. **Asset Code Format**: Asset codes must match `^[A-Z0-9]{1,12}$` pattern
3. **Ledger Sequence**: Must be positive integers

### Business Rules

1. **Fee Validation**: Fees must be non-negative integers
2. **Amount Validation**: Amounts must be non-negative numbers
3. **Operation Count**: Must be between 1 and 100 (Stellar limit)
4. **Balance Format**: Must be valid numbers (no NaN/infinite values)

### Statistical Validation

1. **Amount Outliers**: Uses IQR method with configurable multiplier (default 1.5)
2. **Timestamp Gaps**: Detects gaps larger than threshold (default 60 minutes)
3. **Duplicate Patterns**: Identifies repeated patterns across specified fields

## Error Types

The framework defines specific error types for different validation failures:

### Temporal Errors
- `MISSING_TIMESTAMP`: Timestamp field is missing
- `INVALID_TIMESTAMP_FORMAT`: Invalid timestamp format
- `TIMESTAMP_ORDER_VIOLATION`: Timestamps not monotonically increasing
- `FUTURE_TIMESTAMP`: Timestamp significantly in the future
- `TIMESTAMP_VALIDATION_ERROR`: General timestamp validation error

### Referential Integrity Errors
- `INVALID_ACCOUNT_TYPE`: Account not a string
- `INVALID_ACCOUNT_FORMAT`: Account doesn't match Stellar format
- `INVALID_ASSET_TYPE`: Asset code not a string
- `INVALID_ASSET_FORMAT`: Asset code doesn't match format
- `INVALID_LEDGER_SEQUENCE_TYPE`: Ledger sequence not an integer
- `INVALID_LEDGER_SEQUENCE`: Ledger sequence not positive

### Business Rule Errors
- `INVALID_FEE_TYPE`: Fee not numeric
- `NEGATIVE_FEE`: Fee is negative
- `INVALID_AMOUNT_TYPE`: Amount not numeric
- `NEGATIVE_AMOUNT`: Amount is negative
- `INVALID_OPERATION_COUNT_TYPE`: Operation count not integer
- `INVALID_OPERATION_COUNT`: Operation count out of bounds
- `INVALID_BALANCE_TYPE`: Balance not numeric
- `INVALID_BALANCE_VALUE`: Balance is NaN or infinite

### Statistical Errors
- `AMOUNT_OUTLIERS_DETECTED`: Statistical outliers found in amounts
- `UNUSUAL_TIMESTAMP_GAPS`: Unusual gaps detected in timestamps
- `DUPLICATE_PATTERNS_DETECTED`: Repeated patterns found
- `OUTLIER_DETECTION_ERROR`: Error during outlier detection
- `GAP_DETECTION_ERROR`: Error during gap detection
- `PATTERN_DETECTION_ERROR`: Error during pattern detection

## Configuration

### Temporal Validation
```python
validator = TemporalValidator(timestamp_field="timestamp")  # Custom timestamp field
result = validator.validate_future_timestamps(transactions, tolerance_minutes=5)
```

### Statistical Validation
```python
stat_validator = StatisticalValidator()
result = stat_validator.detect_amount_outliers(amounts, iqr_multiplier=2.0)
result = stat_validator.detect_timestamp_gaps(timestamps, gap_threshold_minutes=120)
result = stat_validator.detect_duplicate_patterns(transactions, ["amount", "source_account"])
```

### Business Rules
```python
biz_validator = BusinessRulesValidator()
# The max operations per transaction is configurable (default 100 for Stellar)
biz_validator.max_operations_per_transaction = 50
```

## Integration with Existing Validation

The data quality validation framework is designed to complement the existing validation infrastructure:

- **Base Validation**: Existing `validator.py` provides corruption detection and basic schema validation
- **Deduplication**: Existing `dedupe.py` provides hash-based duplicate detection
- **Integrity Pipeline**: Existing `integrity.py` combines validation and deduplication
- **Extended Validation**: New `data_quality.py` adds comprehensive domain-specific validation

### Example Integration

```python
from astroml.validation import integrity, data_quality

# Use existing integrity validation
integrity_validator = integrity.IntegrityValidator(required_fields={"id", "source_account"})
integrity_result = integrity_validator.process(transactions)

# Use extended data quality validation
dq_validator = data_quality.DataQualityValidator()
dq_report = dq_validator.validate_batch(transactions)

# Combine results
print(f"Integrity: {integrity_result.is_valid}")
print(f"Data Quality Score: {dq_report.quality_score:.1f}%")
```

## Testing

The framework includes comprehensive test coverage:

### Test Files
- `tests/validation/test_extended_data_quality.py` - Tests for new validation utilities
- `tests/validation/test_data_quality.py` - Enhanced existing tests

### Test Categories
1. **Unit Tests**: Individual validator class tests
2. **Integration Tests**: Comprehensive validator tests
3. **Fixture Tests**: Tests using sample data fixtures
4. **Error Case Tests**: Tests for invalid data scenarios

### Running Tests

```bash
# Run extended data quality tests
python -m pytest tests/validation/test_extended_data_quality.py -v

# Run all validation tests
python -m pytest tests/validation/ -v

# Run specific test class
python -m pytest tests/validation/test_extended_data_quality.py::TestDataQualityValidator -v
```

## Performance Considerations

### Batch Processing
- Validators are designed for efficient batch processing
- Statistical validations require sufficient data for meaningful results
- Large datasets should be processed in manageable chunks

### Memory Usage
- Statistical validators store intermediate results for analysis
- Temporal validators maintain timestamp lists for ordering checks
- Pattern detection uses dictionaries for frequency counting

### Optimization Tips
1. Use appropriate batch sizes for large datasets
2. Configure statistical thresholds based on your data characteristics
3. Select relevant pattern fields for duplicate detection
4. Adjust tolerance parameters for temporal validation

## Extending the Framework

### Adding New Validation Types

1. Create a new validator class following the existing pattern
2. Implement validation methods returning `ValidationResult`
3. Add error types to the appropriate category
4. Update `DataQualityValidator` to include the new validator
5. Add comprehensive tests

### Example Custom Validator

```python
class CustomValidator:
    def validate_custom_rule(self, data: Dict[str, Any]) -> ValidationResult:
        # Implement custom validation logic
        if self.check_condition(data):
            return ValidationResult(is_valid=True, message="Custom rule passed")
        else:
            return ValidationResult(
                is_valid=False,
                error_type="CUSTOM_RULE_VIOLATION",
                message="Custom rule failed"
            )
```

## Best Practices

1. **Layered Validation**: Use multiple validation layers for comprehensive coverage
2. **Error Handling**: Always check validation results before processing
3. **Configuration**: Adjust thresholds and parameters based on your data
4. **Monitoring**: Track quality scores over time to detect data degradation
5. **Testing**: Include validation tests in your CI/CD pipeline

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and paths are correct
2. **Timestamp Format**: Use ISO 8601 format for timestamps
3. **Memory Issues**: Process large datasets in smaller batches
4. **Performance**: Optimize validation parameters for your data size

### Debug Tips

1. Use detailed validation results to identify specific issues
2. Enable logging to track validation progress
3. Test with small, representative datasets first
4. Monitor quality scores to detect trends

## Future Enhancements

Planned improvements to the data quality validation framework:

1. **Machine Learning Validation**: Add ML-based anomaly detection
2. **Real-time Validation**: Support for streaming data validation
3. **Custom Rule Engine**: Allow user-defined validation rules
4. **Performance Optimization**: Parallel processing for large datasets
5. **Enhanced Reporting**: More detailed analytics and visualization
6. **Integration**: Better integration with data pipeline monitoring tools
