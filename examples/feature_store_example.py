#!/usr/bin/env python3
"""
Feature Store Example

This example demonstrates how to use the AstroML Feature Store for
computing, storing, and managing features for machine learning workflows.

This example can be run from any working directory.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

# Add the parent directory to the path to import astroml
# This allows the example to run from any working directory
script_dir = Path(__file__).parent.resolve()
repo_root = script_dir.parent
sys.path.insert(0, str(repo_root))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sample_data():
    """Generate sample transaction data for demonstration."""
    np.random.seed(42)
    
    # Generate sample accounts
    n_accounts = 100
    accounts = [f"account_{i:04d}" for i in range(n_accounts)]
    
    # Generate sample transactions
    n_transactions = 5000
    transactions = []
    
    for i in range(n_transactions):
        # Random timestamp over the last 90 days
        timestamp = datetime.utcnow() - timedelta(
            days=np.random.randint(0, 90),
            hours=np.random.randint(0, 24),
            minutes=np.random.randint(0, 60)
        )
        
        # Random accounts
        src_account = np.random.choice(accounts)
        dst_account = np.random.choice([a for a in accounts if a != src_account])
        
        # Random amount (exponential distribution for realistic amounts)
        amount = np.random.exponential(100)  # Mean of 100 units
        
        # Random asset
        asset = np.random.choice(["XLM", "USD", "EUR", "BTC"], p=[0.5, 0.3, 0.15, 0.05])
        
        transactions.append({
            "entity_id": src_account,  # Source account as entity
            "timestamp": timestamp,
            "amount": amount,
            "src": src_account,
            "dst": dst_account,
            "asset": asset,
            "transaction_type": np.random.choice(["payment", "exchange", "transfer"]),
        })
    
    return pd.DataFrame(transactions)


def custom_balance_computer(data, entity_col, timestamp_col, **kwargs):
    """Custom feature computer for account balance."""
    logger.info("Computing account balance feature")
    
    # Compute total sent and received per account
    sent = data.groupby("src")["amount"].sum()
    received = data.groupby("dst")["amount"].sum()
    
    # Combine sent and received
    all_accounts = set(sent.index) | set(received.index)
    balances = {}
    
    for account in all_accounts:
        sent_amount = sent.get(account, 0)
        received_amount = received.get(account, 0)
        balances[account] = received_amount - sent_amount
    
    result = pd.DataFrame(
        {"account_balance": list(balances.values())},
        index=list(balances.keys())
    )
    
    logger.info(f"Computed balance for {len(result)} accounts")
    return result


def custom_activity_computer(data, entity_col, timestamp_col, **kwargs):
    """Custom feature computer for account activity metrics."""
    logger.info("Computing account activity features")
    
    window_days = kwargs.get("window_days", 30)
    
    # Filter data by time window
    cutoff_time = data[timestamp_col].max() - timedelta(days=window_days)
    recent_data = data[data[timestamp_col] >= cutoff_time]
    
    # Compute activity metrics
    activity_metrics = recent_data.groupby(entity_col).agg({
        "amount": ["count", "sum", "mean", "std"],
        "timestamp": ["min", "max"],
    })
    
    # Flatten column names
    activity_metrics.columns = [
        "transaction_count",
        "total_amount", 
        "avg_amount",
        "std_amount",
        "first_transaction",
        "last_transaction",
    ]
    
    # Fill missing std with 0
    activity_metrics["std_amount"] = activity_metrics["std_amount"].fillna(0)
    
    # Add activity duration
    activity_metrics["activity_duration_days"] = (
        activity_metrics["last_transaction"] - activity_metrics["first_transaction"]
    ).dt.days
    
    logger.info(f"Computed activity metrics for {len(activity_metrics)} accounts")
    return activity_metrics


def custom_asset_diversity_computer(data, entity_col, timestamp_col, **kwargs):
    """Custom feature computer for asset diversity."""
    logger.info("Computing asset diversity feature")
    
    # Count unique assets per account
    asset_diversity = data.groupby(entity_col)["asset"].nunique()
    
    # Compute asset distribution entropy
    def entropy(series):
        """Calculate Shannon entropy."""
        counts = series.value_counts(normalize=True)
        return -np.sum(counts * np.log2(counts + 1e-10))
    
    asset_entropy = data.groupby(entity_col)["asset"].apply(entropy)
    
    result = pd.DataFrame({
        "asset_diversity": asset_diversity,
        "asset_entropy": asset_entropy,
    })
    
    logger.info(f"Computed asset diversity for {len(result)} accounts")
    return result


def main():
    """Main example function."""
    print("🚀 AstroML Feature Store Example")
    print("=" * 50)
    
    # Create temporary directory for the example
    temp_dir = tempfile.mkdtemp()
    store_path = Path(temp_dir) / "example_feature_store"
    
    try:
        # Import Feature Store components
        from astroml.features import create_feature_store
        from astroml.features.feature_store import FeatureType
        
        print(f"📁 Using temporary store path: {store_path}")
        
        # 1. Create Feature Store
        print("\n1️⃣ Creating Feature Store...")
        store = create_feature_store(str(store_path))
        print("✅ Feature Store created successfully")
        
        # 2. Generate sample data
        print("\n2️⃣ Generating sample transaction data...")
        data = generate_sample_data()
        print(f"✅ Generated {len(data)} transactions for {data['entity_id'].nunique()} accounts")
        print(f"   Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
        print(f"   Assets: {', '.join(data['asset'].unique())}")
        
        # 3. Register custom features
        print("\n3️⃣ Registering custom features...")
        
        # Register balance feature
        balance_def = store.register_feature(
            name="account_balance",
            computer=custom_balance_computer,
            description="Account balance computed from transaction inflows and outflows",
            feature_type=FeatureType.NUMERIC,
            tags=["balance", "financial", "basic"],
            owner="example_team",
        )
        print(f"✅ Registered feature: {balance_def.name}")
        
        # Register activity feature
        activity_def = store.register_feature(
            name="account_activity",
            computer=custom_activity_computer,
            description="Account activity metrics including transaction counts and amounts",
            feature_type=FeatureType.TIME_SERIES,
            tags=["activity", "behavior", "engagement"],
            owner="example_team",
            parameters={"window_days": 30},
        )
        print(f"✅ Registered feature: {activity_def.name}")
        
        # Register asset diversity feature
        diversity_def = store.register_feature(
            name="asset_diversity",
            computer=custom_asset_diversity_computer,
            description="Asset diversity and entropy metrics",
            feature_type=FeatureType.NUMERIC,
            tags=["diversity", "risk", "portfolio"],
            owner="example_team",
        )
        print(f"✅ Registered feature: {diversity_def.name}")
        
        # 4. Compute and store features
        print("\n4️⃣ Computing and storing features...")
        
        # Compute balance feature
        print("   Computing account balance...")
        balance_values = store.compute_and_store(
            feature_name="account_balance",
            data=data,
            entity_col="entity_id",
            timestamp_col="timestamp",
        )
        print(f"   ✅ Computed balance for {len(balance_values)} accounts")
        
        # Compute activity feature
        print("   Computing account activity...")
        activity_values = store.compute_and_store(
            feature_name="account_activity",
            data=data,
            entity_col="entity_id", 
            timestamp_col="timestamp",
            window_days=30,
        )
        print(f"   ✅ Computed activity for {len(activity_values)} accounts")
        
        # Compute asset diversity feature
        print("   Computing asset diversity...")
        diversity_values = store.compute_and_store(
            feature_name="asset_diversity",
            data=data,
            entity_col="entity_id",
            timestamp_col="timestamp",
        )
        print(f"   ✅ Computed diversity for {len(diversity_values)} accounts")
        
        # 5. Create feature sets
        print("\n5️⃣ Creating feature sets...")
        
        # Create basic feature set
        basic_features = store.create_feature_set(
            name="basic_account_features",
            feature_names=["account_balance", "account_activity"],
            description="Basic account features for general analysis",
            entity_type="account",
        )
        print(f"✅ Created feature set: {basic_features.name} with {len(basic_features.feature_ids)} features")
        
        # Create risk feature set
        risk_features = store.create_feature_set(
            name="risk_assessment_features",
            feature_names=["account_balance", "account_activity", "asset_diversity"],
            description="Features for risk assessment and fraud detection",
            entity_type="account",
        )
        print(f"✅ Created feature set: {risk_features.name} with {len(risk_features.feature_ids)} features")
        
        # 6. Retrieve and analyze features
        print("\n6️⃣ Retrieving and analyzing features...")
        
        # Get sample accounts
        sample_accounts = data["entity_id"].unique()[:10]
        print(f"   Analyzing {len(sample_accounts)} sample accounts")
        
        # Retrieve features for sample accounts
        sample_features = store.get_features_for_entities(
            feature_names=["account_balance", "account_activity", "asset_diversity"],
            entity_ids=sample_accounts.tolist(),
        )
        
        print("   Sample feature values:")
        print(sample_features.round(2).head())
        
        # Feature statistics
        print("\n   Feature Statistics:")
        print(f"   Account Balance - Mean: {balance_values['account_balance'].mean():.2f}, "
              f"Std: {balance_values['account_balance'].std():.2f}")
        print(f"   Transaction Count - Mean: {activity_values['transaction_count'].mean():.2f}, "
              f"Std: {activity_values['transaction_count'].std():.2f}")
        print(f"   Asset Diversity - Mean: {diversity_values['asset_diversity'].mean():.2f}, "
              f"Std: {diversity_values['asset_diversity'].std():.2f}")
        
        # 7. Feature discovery
        print("\n7️⃣ Discovering available features...")
        
        all_features = store.list_features()
        print(f"   Total features available: {len(all_features)}")
        
        print("\n   Available features:")
        for feature in all_features:
            print(f"   - {feature.name}: {feature.description}")
            print(f"     Type: {feature.feature_type.value}, Tags: {', '.join(feature.tags)}")
        
        # 8. Cache performance
        print("\n8️⃣ Testing cache performance...")
        
        # First retrieval (cache miss)
        import time
        start_time = time.time()
        features_1 = store.get_feature("account_balance")
        first_time = time.time() - start_time
        
        # Second retrieval (cache hit)
        start_time = time.time()
        features_2 = store.get_feature("account_balance")
        second_time = time.time() - start_time
        
        print(f"   First retrieval (cache miss): {first_time:.4f}s")
        print(f"   Second retrieval (cache hit): {second_time:.4f}s")
        print(f"   Cache speedup: {first_time/second_time:.1f}x")
        
        # Cache statistics
        cache_stats = store.cache.get_stats()
        print(f"   Cache hit rate: {cache_stats['hit_rate']:.2%}")
        print(f"   Cache size: {cache_stats['size']}")
        
        # 9. Feature transformations
        print("\n9️⃣ Demonstrating feature transformations...")
        
        try:
            from astroml.features.feature_transformers import (
                create_feature_transformer,
                TransformationType,
                apply_standard_scaling,
            )
            
            # Combine features for transformation
            combined_features = store.get_features_for_entities(
                feature_names=["account_balance", "account_activity"],
                entity_ids=balance_values.index.tolist(),
            )
            
            # Apply standard scaling
            scaled_features, transformer = apply_standard_scaling(
                combined_features,
                ["account_balance", "transaction_count", "total_amount"],
            )
            
            print("   Applied standard scaling to features")
            print("   Scaled features summary:")
            print(scaled_features.describe().round(2))
            
        except ImportError:
            print("   ⚠️  Feature transformers not available")
        
        # 10. Feature versioning (if available)
        print("\n🔟 Feature versioning...")
        
        try:
            from astroml.features.feature_versioning import create_version_manager, VersionStatus
            
            version_manager = create_version_manager(str(store_path / "versions"))
            
            # Create a version for our balance feature
            version = version_manager.create_version(
                feature_name="account_balance",
                code=custom_balance_computer.__code__.co_code,
                parameters={},
                data_schema={"entity_id": "string", "amount": "float"},
                description="Initial version of account balance feature",
                created_by="example_script",
            )
            
            print(f"   Created version {version.version} for account_balance")
            
            # Update status
            version_manager.update_version_status(
                version_id=version.version_id,
                status=VersionStatus.APPROVED,
                updated_by="example_script",
            )
            
            print(f"   Updated version status to: {VersionStatus.APPROVED.value}")
            
        except ImportError:
            print("   ⚠️  Feature versioning not available")
        
        print("\n🎉 Feature Store example completed successfully!")
        print(f"   📊 Processed {len(data)} transactions")
        print(f"   🔧 Computed {len(all_features)} features")
        print(f"   📦 Created {len(store.list_features())} feature sets")
        print(f"   💾 Stored in: {store_path}")
        
        # Show some example use cases
        print("\n💡 Example Use Cases:")
        print("   1. Machine Learning: Use stored features for model training")
        print("   2. Real-time Scoring: Retrieve features for online predictions")
        print("   3. Analytics: Analyze feature distributions and trends")
        print("   4. Monitoring: Track feature quality and drift over time")
        print("   5. Collaboration: Share features across teams and projects")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\n🧹 Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main()
