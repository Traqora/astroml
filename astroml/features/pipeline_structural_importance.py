"""Pipeline integration for structural importance metrics.

This module provides pipeline step functions to calculate structural importance
metrics for account nodes in the AstroML processing pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Any
import pandas as pd
from sqlalchemy.orm import Session

from astroml.db.schema import Operation, NormalizedTransaction, Transaction
from astroml.features.structural_importance import compute_structural_importance_metrics

logger = logging.getLogger(__name__)


class StructuralImportancePipeline:
    """Pipeline step for computing structural importance metrics on account nodes.
    
    This class provides methods to integrate structural importance calculations
    into the AstroML data processing pipeline. It can work with both raw operations
    and normalized transactions to compute various centrality and importance measures.
    """
    
    def __init__(
        self,
        include_betweenness: bool = True,
        include_closeness: bool = True,
        include_eigenvector: bool = False,
        pagerank_sample_size: Optional[int] = None,
        betweenness_sample_size: Optional[int] = None,
        batch_size: int = 10000
    ):
        """Initialize the structural importance pipeline step.
        
        Args:
            include_betweenness: Whether to compute betweenness centrality.
            include_closeness: Whether to compute closeness centrality.
            include_eigenvector: Whether to compute eigenvector centrality.
            pagerank_sample_size: Sample size for PageRank approximation on large graphs.
            betweenness_sample_size: Sample size for betweenness approximation.
            batch_size: Number of transactions to process in each batch.
        """
        self.include_betweenness = include_betweenness
        self.include_closeness = include_closeness
        self.include_eigenvector = include_eigenvector
        self.pagerank_sample_size = pagerank_sample_size
        self.betweenness_sample_size = betweenness_sample_size
        self.batch_size = batch_size
    
    def process_operations(
        self,
        session: Session,
        start_ledger: Optional[int] = None,
        end_ledger: Optional[int] = None,
        account_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process operations from the database to compute structural importance.
        
        Args:
            session: SQLAlchemy database session.
            start_ledger: Optional starting ledger sequence (inclusive).
            end_ledger: Optional ending ledger sequence (inclusive).
            account_filter: Optional list of account IDs to include.
            
        Returns:
            DataFrame with structural importance metrics indexed by account_id.
        """
        logger.info("Starting structural importance computation from operations")
        
        # Build query
        query = session.query(Operation).order_by(Operation.id)
        
        if start_ledger is not None:
            query = query.join(Operation.transaction).filter(
                Transaction.ledger_sequence >= start_ledger
            )
        
        if end_ledger is not None:
            query = query.join(Operation.transaction).filter(
                Transaction.ledger_sequence <= end_ledger
            )
        
        if account_filter is not None:
            query = query.filter(
                (Operation.source_account.in_(account_filter)) |
                (Operation.destination_account.in_(account_filter))
            )
        
        # Process in batches using keyset pagination
        edges = []
        total_processed = 0
        last_id = None
        
        while True:
            batch_query = query
            if last_id is not None:
                batch_query = batch_query.filter(Operation.id > last_id)
            batch = batch_query.limit(self.batch_size).all()
            
            if not batch:
                break
                
            for op in batch:
                if op.source_account and op.destination_account:
                    edges.append({
                        'src': op.source_account,
                        'dst': op.destination_account,
                        'amount': float(op.amount) if op.amount else 0.0,
                        'timestamp': op.created_at.timestamp()
                    })
            
            total_processed += len(batch)
            if total_processed % (self.batch_size * 5) == 0:
                logger.info(f"Processed {total_processed} operations")
                
            last_id = batch[-1].id
        
        logger.info(f"Extracted {len(edges)} edges from {total_processed} operations")
        
        # Compute structural importance metrics
        return compute_structural_importance_metrics(
            edges=edges,
            nodes=account_filter,
            include_betweenness=self.include_betweenness,
            include_closeness=self.include_closeness,
            include_eigenvector=self.include_eigenvector,
            pagerank_sample_size=self.pagerank_sample_size,
            betweenness_sample_size=self.betweenness_sample_size
        )
    
    def process_normalized_transactions(
        self,
        session: Session,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        account_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process normalized transactions to compute structural importance.
        
        Args:
            session: SQLAlchemy database session.
            start_time: Optional start time (ISO format string).
            end_time: Optional end time (ISO format string).
            account_filter: Optional list of account IDs to include.
            
        Returns:
            DataFrame with structural importance metrics indexed by account_id.
        """
        logger.info("Starting structural importance computation from normalized transactions")
        
        # Build query
        query = session.query(NormalizedTransaction).order_by(NormalizedTransaction.id)
        
        if start_time is not None:
            query = query.filter(NormalizedTransaction.timestamp >= start_time)
        
        if end_time is not None:
            query = query.filter(NormalizedTransaction.timestamp <= end_time)
        
        if account_filter is not None:
            query = query.filter(
                (NormalizedTransaction.sender.in_(account_filter)) |
                (NormalizedTransaction.receiver.in_(account_filter))
            )
        
        # Process in batches using keyset pagination
        edges = []
        total_processed = 0
        last_id = None
        
        while True:
            batch_query = query
            if last_id is not None:
                batch_query = batch_query.filter(NormalizedTransaction.id > last_id)
            batch = batch_query.limit(self.batch_size).all()
            
            if not batch:
                break
                
            for tx in batch:
                if tx.sender and tx.receiver:
                    edges.append({
                        'src': tx.sender,
                        'dst': tx.receiver,
                        'amount': float(tx.amount) if tx.amount else 0.0,
                        'timestamp': tx.timestamp.timestamp()
                    })
            
            total_processed += len(batch)
            if total_processed % (self.batch_size * 5) == 0:
                logger.info(f"Processed {total_processed} normalized transactions")
                
            last_id = batch[-1].id
        
        logger.info(f"Extracted {len(edges)} edges from {total_processed} normalized transactions")
        
        # Compute structural importance metrics
        return compute_structural_importance_metrics(
            edges=edges,
            nodes=account_filter,
            include_betweenness=self.include_betweenness,
            include_closeness=self.include_closeness,
            include_eigenvector=self.include_eigenvector,
            pagerank_sample_size=self.pagerank_sample_size,
            betweenness_sample_size=self.betweenness_sample_size
        )
    
    def process_edge_list(
        self,
        edges: Iterable[Dict[str, Any]],
        account_filter: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Process a list of edges to compute structural importance.
        
        Args:
            edges: Iterable of edge dictionaries with 'src', 'dst', 'amount', 'timestamp'.
            account_filter: Optional list of account IDs to include.
            
        Returns:
            DataFrame with structural importance metrics indexed by account_id.
        """
        logger.info(f"Processing {len(list(edges)) if hasattr(edges, '__len__') else 'unknown'} edges")
        
        return compute_structural_importance_metrics(
            edges=edges,
            nodes=account_filter,
            include_betweenness=self.include_betweenness,
            include_closeness=self.include_closeness,
            include_eigenvector=self.include_eigenvector,
            pagerank_sample_size=self.pagerank_sample_size,
            betweenness_sample_size=self.betweenness_sample_size
        )
    
    def get_summary_statistics(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics for the computed metrics.
        
        Args:
            metrics_df: DataFrame with structural importance metrics.
            
        Returns:
            Dictionary with summary statistics for each metric.
        """
        summary = {
            'total_accounts': len(metrics_df),
            'metrics': {}
        }
        
        for column in metrics_df.columns:
            series = metrics_df[column]
            summary['metrics'][column] = {
                'mean': float(series.mean()),
                'std': float(series.std()),
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
                'non_zero_count': int((series != 0).sum()),
                'top_accounts': series.nlargest(10).to_dict()
            }
        
        return summary


def run_structural_importance_pipeline(
    session: Session,
    source: str = 'operations',
    start_ledger: Optional[int] = None,
    end_ledger: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    account_filter: Optional[List[str]] = None,
    **pipeline_kwargs
) -> pd.DataFrame:
    """Convenience function to run the structural importance pipeline.
    
    Args:
        session: SQLAlchemy database session.
        source: Data source ('operations' or 'normalized').
        start_ledger: Starting ledger sequence (for operations source).
        end_ledger: Ending ledger sequence (for operations source).
        start_time: Start time (for normalized source).
        end_time: End time (for normalized source).
        account_filter: Optional list of account IDs to include.
        **pipeline_kwargs: Additional arguments for StructuralImportancePipeline.
        
    Returns:
        DataFrame with structural importance metrics.
    """
    pipeline = StructuralImportancePipeline(**pipeline_kwargs)
    
    if source == 'operations':
        return pipeline.process_operations(
            session=session,
            start_ledger=start_ledger,
            end_ledger=end_ledger,
            account_filter=account_filter
        )
    elif source == 'normalized':
        return pipeline.process_normalized_transactions(
            session=session,
            start_time=start_time,
            end_time=end_time,
            account_filter=account_filter
        )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'operations' or 'normalized'.")
