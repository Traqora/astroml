"""Transaction graph construction module.

Constructs directed graphs where nodes represent accounts and edges represent
transactions between them. Supports weighted edges, multi-asset transactions,
and export to NetworkX format.
"""
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict


class TransactionGraph:
    """Directed graph representation of account transactions.
    
    Nodes represent accounts, edges represent transactions with weights
    corresponding to transaction amounts. Supports multiple assets.
    """
    
    def __init__(self):
        """Initialize an empty transaction graph."""
        self.nodes = set()
        self.edges = defaultdict(lambda: defaultdict(list))
        self._asset_edges = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    def add_transaction(
        self,
        from_account: str,
        to_account: str,
        amount: float,
        asset: str = "default",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a transaction edge to the graph.
        
        Args:
            from_account: Source account identifier
            to_account: Destination account identifier
            amount: Transaction amount (weight)
            asset: Asset type (e.g., 'USD', 'BTC', 'ETH')
            metadata: Optional transaction metadata
        """
        self.nodes.add(from_account)
        self.nodes.add(to_account)
        
        transaction = {
            "amount": amount,
            "asset": asset,
            "metadata": metadata or {}
        }
        
        self.edges[from_account][to_account].append(transaction)
        self._asset_edges[asset][from_account][to_account].append(transaction)
    
    def get_edge_weight(
        self,
        from_account: str,
        to_account: str,
        asset: Optional[str] = None,
        aggregation: str = "sum"
    ) -> float:
        """Get aggregated weight for an edge.
        
        Args:
            from_account: Source account
            to_account: Destination account
            asset: Optional asset filter (None = all assets)
            aggregation: Aggregation method ('sum', 'mean', 'count', 'max', 'min')
            
        Returns:
            Aggregated edge weight
        """
        if asset:
            transactions = self._asset_edges[asset][from_account][to_account]
        else:
            transactions = self.edges[from_account][to_account]
        
        if not transactions:
            return 0.0
        
        amounts = [t["amount"] for t in transactions]
        
        if aggregation == "sum":
            return sum(amounts)
        elif aggregation == "mean":
            return sum(amounts) / len(amounts)
        elif aggregation == "count":
            return float(len(amounts))
        elif aggregation == "max":
            return max(amounts)
        elif aggregation == "min":
            return min(amounts)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    def get_assets(self) -> List[str]:
        """Get list of all assets in the graph.
        
        Returns:
            List of asset identifiers
        """
        return list(self._asset_edges.keys())
    
    def get_transactions(
        self,
        from_account: Optional[str] = None,
        to_account: Optional[str] = None,
        asset: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get transactions matching filters.
        
        Args:
            from_account: Optional source account filter
            to_account: Optional destination account filter
            asset: Optional asset filter
            
        Returns:
            List of matching transactions with from/to account info
        """
        results = []
        
        if asset:
            edge_dict = self._asset_edges[asset]
        else:
            edge_dict = self.edges
        
        for src in edge_dict:
            if from_account and src != from_account:
                continue
            for dst in edge_dict[src]:
                if to_account and dst != to_account:
                    continue
                for txn in edge_dict[src][dst]:
                    results.append({
                        "from": src,
                        "to": dst,
                        **txn
                    })
        
        return results
    
    def to_networkx(
        self,
        asset: Optional[str] = None,
        aggregation: str = "sum",
        include_metadata: bool = False
    ):
        """Export graph to NetworkX DiGraph format.
        
        Args:
            asset: Optional asset filter (None = all assets)
            aggregation: Weight aggregation method
            include_metadata: Include transaction metadata as edge attributes
            
        Returns:
            NetworkX DiGraph object
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "NetworkX is required for graph export. "
                "Install it with: pip install networkx"
            )
        
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(self.nodes)
        
        # Add edges with weights
        if asset:
            edge_dict = self._asset_edges[asset]
        else:
            edge_dict = self.edges
        
        for src in edge_dict:
            for dst in edge_dict[src]:
                weight = self.get_edge_weight(src, dst, asset, aggregation)
                edge_attrs = {"weight": weight}
                
                if include_metadata:
                    transactions = edge_dict[src][dst]
                    edge_attrs["transaction_count"] = len(transactions)
                    edge_attrs["transactions"] = transactions
                
                G.add_edge(src, dst, **edge_attrs)
        
        return G
    
    def summary(self) -> Dict[str, Any]:
        """Get graph summary statistics.
        
        Returns:
            Dictionary with graph statistics
        """
        total_transactions = sum(
            len(txns)
            for src_edges in self.edges.values()
            for txns in src_edges.values()
        )
        
        asset_stats = {}
        for asset in self._asset_edges:
            asset_txn_count = sum(
                len(txns)
                for src_edges in self._asset_edges[asset].values()
                for txns in src_edges.values()
            )
            asset_stats[asset] = asset_txn_count
        
        return {
            "node_count": len(self.nodes),
            "edge_count": sum(
                len(dests) for dests in self.edges.values()
            ),
            "transaction_count": total_transactions,
            "asset_count": len(self._asset_edges),
            "assets": asset_stats
        }
