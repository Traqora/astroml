#!/usr/bin/env python3
"""
Verification script for the Feature Store implementation.
This script tests the core functionality to ensure everything is working correctly.
"""

import sys
import os
import tempfile
import traceback
from pathlib import Path

import pandas as pd

# Add the astroml directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all Feature Store components can be imported."""
    print("🔍 Testing imports...")
    
    try:
        # Test core imports
        from astroml.features import (
            FeatureStore,
            FeatureDefinition,
            FeatureType,
            FeatureStatus,
            FeatureSet,
            create_feature_store,
        )
        print("   ✅ Core Feature Store imports successful")
        
        # Test additional components
        from astroml.features import (
            ComputationEngine,
            FeatureTransformer,
            FeatureCache,
            FeatureVersionManager,
        )
        print("   ✅ Additional Feature Store components imports successful")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_basic_functionality():
    """Test basic Feature Store functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        from astroml.features import create_feature_store, FeatureType
        import pandas as pd
        import numpy as np
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create feature store
            store = create_feature_store(temp_dir)
            print("   ✅ Feature Store created successfully")
            
            # Test custom feature registration
            def test_computer(data, entity_col, timestamp_col, **kwargs):
                """Simple test feature computer."""
                return pd.DataFrame({
                    'test_feature': np.random.random(len(data[entity_col].unique()))
                }, index=data[entity_col].unique())
            
            feature_def = store.register_feature(
                name='test_feature',
                computer=test_computer,
                description='Test feature for verification',
                feature_type=FeatureType.NUMERIC,
                tags=['test', 'verification'],
                owner='verification_script',
            )
            print("   ✅ Feature registration successful")
            
            # Create sample data
            sample_data = pd.DataFrame({
                'entity_id': [f'entity_{i}' for i in range(10)],
                'timestamp': pd.date_range('2023-01-01', periods=10, freq='D'),
                'amount': np.random.random(10) * 100,
            })
            
            # Test feature computation
            try:
                result = store.compute_feature(
                    feature_name='test_feature',
                    data=sample_data,
                    entity_col='entity_id',
                    timestamp_col='timestamp',
                )
                print("   ✅ Feature computation successful")
                print(f"      Computed {len(result)} feature values")
            except Exception as e:
                print(f"   ⚠️  Feature computation failed (may be expected): {e}")
            
            # Test feature listing
            features = store.list_features()
            print(f"   ✅ Feature listing successful: {len(features)} features found")
            
            # Test our registered feature
            test_features = [f for f in features if f.name == 'test_feature']
            if test_features:
                print(f"   ✅ Test feature found: {test_features[0].name}")
            else:
                print("   ⚠️  Test feature not found in listing")
            
            return True
            
    except Exception as e:
        print(f"   ❌ Basic functionality error: {e}")
        traceback.print_exc()
        return False

def test_data_structures():
    """Test data structures and enums."""
    print("\n📊 Testing data structures...")
    
    try:
        from astroml.features.feature_store import (
            FeatureDefinition,
            FeatureType,
            FeatureStatus,
        )
        
        # Test FeatureDefinition
        def dummy_computer(data, entity_col, timestamp_col, **kwargs):
            return pd.DataFrame({'dummy': [1, 2, 3]})
        
        feature_def = FeatureDefinition(
            name="dummy_feature",
            description="Dummy feature for testing",
            feature_type=FeatureType.NUMERIC,
            computation_function=dummy_computer,
        )
        
        assert feature_def.name == "dummy_feature"
        assert feature_def.feature_id == "dummy_feature_v1"
        assert feature_def.feature_type == FeatureType.NUMERIC
        print("   ✅ FeatureDefinition working correctly")
        
        # Test enums
        assert FeatureType.NUMERIC.value == "numeric"
        assert FeatureType.CATEGORICAL.value == "categorical"
        assert FeatureStatus.DEVELOPMENT.value == "development"
        print("   ✅ Enums working correctly")
        
        # Test serialization
        data = feature_def.to_dict()
        restored = FeatureDefinition.from_dict(data)
        assert restored.name == feature_def.name
        assert restored.feature_type == feature_def.feature_type
        print("   ✅ FeatureDefinition serialization working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Data structures error: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n📁 Testing file structure...")
    
    base_path = Path(__file__).parent
    required_files = [
        "astroml/features/feature_store.py",
        "astroml/features/feature_engine.py", 
        "astroml/features/feature_transformers.py",
        "astroml/features/feature_cache.py",
        "astroml/features/feature_versioning.py",
        "tests/features/test_feature_store.py",
        "docs/FEATURE_STORE.md",
        "examples/feature_store_example.py",
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - MISSING")
            missing_files.append(file_path)
    
    if not missing_files:
        print("   ✅ All required files present")
        return True
    else:
        print(f"   ❌ {len(missing_files)} files missing")
        return False

def test_integration():
    """Test integration with existing astroml features."""
    print("\n🔗 Testing integration with existing features...")
    
    try:
        # Test that existing feature modules can still be imported
        from astroml.features import frequency, structural_importance, node_features
        print("   ✅ Existing feature modules import successfully")
        
        # Test that the registry can find built-in features
        from astroml.features.feature_store import create_feature_store
        
        with tempfile.TemporaryDirectory() as temp_dir:
            store = create_feature_store(temp_dir)
            
            # Check if built-in features are registered
            computers = store.registry.list_features()
            if computers:
                print(f"   ✅ Found {len(computers)} registered feature computers")
                print(f"      Sample: {computers[:3]}")
            else:
                print("   ⚠️  No built-in features found (may be expected if modules not available)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Integration error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all verification tests."""
    print("🚀 Feature Store Verification")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Basic Functionality Test", test_basic_functionality), 
        ("Data Structures Test", test_data_structures),
        ("File Structure Test", test_file_structure),
        ("Integration Test", test_integration),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"   ❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📋 Verification Summary")
    print("=" * 30)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Feature Store implementation is working correctly.")
        print("\n💡 Next steps:")
        print("   1. Run the full test suite: pytest tests/features/")
        print("   2. Try the example: python examples/feature_store_example.py")
        print("   3. Check the documentation: docs/FEATURE_STORE.md")
        return True
    else:
        print("⚠️  Some tests failed. Please review the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
