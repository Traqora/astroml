#!/usr/bin/env python3
"""Simple test script to verify data quality validation imports."""

import sys
import os

# Add the astroml directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

try:
    # Test importing the data quality module
    from astroml.validation.data_quality import (
        DataQualityValidator,
        TemporalValidator,
        ReferentialIntegrityValidator,
        BusinessRulesValidator,
        StatisticalValidator,
        validate_data_quality,
        check_temporal_consistency,
        check_referential_integrity,
    )
    
    print("✓ Successfully imported data quality validation components")
    
    # Test basic functionality
    validator = DataQualityValidator()
    print("✓ Successfully created DataQualityValidator instance")
    
    # Test with sample data
    from datetime import datetime, timedelta
    base_time = datetime.utcnow()
    
    sample_transactions = [
        {
            "id": "tx_1",
            "timestamp": (base_time + timedelta(hours=1)).isoformat(),
            "source_account": "GABCD1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890",
            "asset_code": "XLM",
            "ledger_sequence": 123,
            "fee": 100,
            "amount": 100.0,
            "operation_count": 1,
        }
    ]
    
    report = validator.validate_batch(sample_transactions)
    print(f"✓ Successfully validated sample transactions: {report.total_records} records")
    print(f"✓ Quality score: {report.quality_score:.1f}%")
    
    # Test convenience functions
    temporal_results = check_temporal_consistency(sample_transactions)
    referential_results = check_referential_integrity(sample_transactions)
    print(f"✓ Temporal consistency checks: {len(temporal_results)} results")
    print(f"✓ Referential integrity checks: {len(referential_results)} results")
    
    print("\n🎉 All data quality validation tests passed!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
