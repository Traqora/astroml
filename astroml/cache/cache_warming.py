"""Cache warming utilities for Redis cache.

Issue #330: Cache warming on startup for hot data paths.
Pre-loads frequently accessed data into Redis cache to improve performance.
"""

from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CacheWarmer:
    """Cache warming service for pre-loading hot data paths."""
    
    def __init__(self, cache_client=None):
        """Initialize cache warmer.
        
        Args:
            cache_client: Optional RedisCache instance. If None, creates new instance.
        """
        try:
            from astroml.cache.redis_cache import RedisCache
            self.cache = cache_client or RedisCache()
            self.available = True
        except ImportError:
            self.cache = None
            self.available = False
            logger.warning("Redis cache not available for warming")
    
    def warm_account_summaries(
        self, 
        account_ids: List[str], 
        db_session=None
    ) -> Dict[str, bool]:
        """Warm cache for account summaries (fraud and loyalty data).
        
        Args:
            account_ids: List of account public keys to warm
            db_session: Optional database session for fetching data
            
        Returns:
            Dictionary mapping account_id to success status
        """
        if not self.available:
            logger.warning("Cache warming skipped - Redis not available")
            return {acc_id: False for acc_id in account_ids}
        
        results = {}
        success_count = 0
        
        for account_id in account_ids:
            try:
                # Warm fraud summary
                fraud_key = f"account:fraud-summary:{account_id}"
                if not self.cache.exists(fraud_key):
                    # In production, fetch from DB and cache
                    # For now, we'll mark as warmed if we can connect
                    self.cache.set(fraud_key, {"account_id": account_id, "total_alerts": 0}, ttl_seconds=300)
                
                # Warm loyalty summary
                loyalty_key = f"account:loyalty:{account_id}"
                if not self.cache.exists(loyalty_key):
                    self.cache.set(
                        loyalty_key, 
                        {"account_id": account_id, "points_balance": 0, "tier_id": "bronze"},
                        ttl_seconds=300
                    )
                
                results[account_id] = True
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for account {account_id}: {e}")
                results[account_id] = False
        
        logger.info(f"Cache warming completed: {success_count}/{len(account_ids)} accounts warmed")
        return results
    
    def warm_feature_cache(
        self, 
        feature_names: List[str],
        ttl_seconds: int = 900
    ) -> Dict[str, bool]:
        """Warm cache for frequently accessed features.
        
        Args:
            feature_names: List of feature names to warm
            ttl_seconds: TTL for cached features (default 15 minutes)
            
        Returns:
            Dictionary mapping feature_name to success status
        """
        if not self.available:
            return {name: False for name in feature_names}
        
        results = {}
        success_count = 0
        
        for feature_name in feature_names:
            try:
                feature_key = f"feature:{feature_name}"
                if not self.cache.exists(feature_key):
                    # Placeholder - in production, fetch actual feature data
                    self.cache.set(feature_key, {"name": feature_name, "cached_at": datetime.utcnow().isoformat()}, ttl_seconds)
                
                results[feature_name] = True
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for feature {feature_name}: {e}")
                results[feature_name] = False
        
        logger.info(f"Feature cache warming completed: {success_count}/{len(feature_names)} features warmed")
        return results
    
    def warm_graph_snapshots(
        self,
        snapshot_ids: List[str],
        ttl_seconds: int = 3600
    ) -> Dict[str, bool]:
        """Warm cache for graph snapshots.
        
        Args:
            snapshot_ids: List of graph snapshot IDs to warm
            ttl_seconds: TTL for cached snapshots (default 1 hour)
            
        Returns:
            Dictionary mapping snapshot_id to success status
        """
        if not self.available:
            return {sid: False for sid in snapshot_ids}
        
        results = {}
        success_count = 0
        
        for snapshot_id in snapshot_ids:
            try:
                snapshot_key = f"graph:snapshot:{snapshot_id}"
                if not self.cache.exists(snapshot_key):
                    # Placeholder - in production, fetch actual graph data
                    self.cache.set(snapshot_key, {"snapshot_id": snapshot_id, "cached_at": datetime.utcnow().isoformat()}, ttl_seconds)
                
                results[snapshot_id] = True
                success_count += 1
            except Exception as e:
                logger.warning(f"Failed to warm cache for snapshot {snapshot_id}: {e}")
                results[snapshot_id] = False
        
        logger.info(f"Graph snapshot warming completed: {success_count}/{len(snapshot_ids)} snapshots warmed")
        return results
    
    def warm_hot_paths(
        self,
        top_accounts: Optional[List[str]] = None,
        top_features: Optional[List[str]] = None,
        top_snapshots: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Warm all hot data paths based on usage patterns.
        
        Args:
            top_accounts: Top N most frequently accessed accounts
            top_features: Top N most frequently accessed features
            top_snapshots: Top N most frequently accessed graph snapshots
            
        Returns:
            Summary of warming results
        """
        logger.info("Starting cache warming for hot data paths")
        
        results = {
            "accounts": {},
            "features": {},
            "snapshots": {},
            "total_warmed": 0,
            "total_failed": 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if top_accounts:
            account_results = self.warm_account_summaries(top_accounts)
            results["accounts"] = account_results
            results["total_warmed"] += sum(1 for v in account_results.values() if v)
            results["total_failed"] += sum(1 for v in account_results.values() if not v)
        
        if top_features:
            feature_results = self.warm_feature_cache(top_features)
            results["features"] = feature_results
            results["total_warmed"] += sum(1 for v in feature_results.values() if v)
            results["total_failed"] += sum(1 for v in feature_results.values() if not v)
        
        if top_snapshots:
            snapshot_results = self.warm_graph_snapshots(top_snapshots)
            results["snapshots"] = snapshot_results
            results["total_warmed"] += sum(1 for v in snapshot_results.values() if v)
            results["total_failed"] += sum(1 for v in snapshot_results.values() if not v)
        
        logger.info(f"Cache warming completed: {results['total_warmed']} warmed, {results['total_failed']} failed")
        return results


def warm_cache_on_startup(
    top_n_accounts: int = 100,
    top_n_features: int = 50,
    top_n_snapshots: int = 10
) -> Dict[str, Any]:
    """Convenience function to warm cache on application startup.
    
    Issue #330: Cache warming on startup.
    
    Args:
        top_n_accounts: Number of top accounts to warm
        top_n_features: Number of top features to warm
        top_n_snapshots: Number of top graph snapshots to warm
        
    Returns:
        Summary of warming results
    """
    warmer = CacheWarmer()
    
    # In production, these would be fetched from analytics/usage metrics
    # For now, use placeholder lists
    top_accounts = [f"account_{i}" for i in range(top_n_accounts)]
    top_features = [f"feature_{i}" for i in range(top_n_features)]
    top_snapshots = [f"snapshot_{i}" for i in range(top_n_snapshots)]
    
    return warmer.warm_hot_paths(top_accounts, top_features, top_snapshots)


__all__ = ["CacheWarmer", "warm_cache_on_startup"]
