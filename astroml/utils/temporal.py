import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import math


class TemporalDataProcessor:
    """Utilities for processing temporal graph data."""
    
    def __init__(self, time_window: Optional[int] = None):
        self.time_window = time_window
        self.reference_time = None
    
    def normalize_timestamps(
        self, 
        timestamps: Union[List[datetime], np.ndarray, torch.Tensor],
        reference_time: Optional[datetime] = None
    ) -> torch.Tensor:
        """Normalize timestamps to relative values.
        
        Args:
            timestamps: List or array of timestamps
            reference_time: Reference time for normalization
            
        Returns:
            Normalized timestamps as tensor
        """
        if reference_time is None:
            reference_time = min(timestamps) if isinstance(timestamps, list) else timestamps.min()
        
        if isinstance(timestamps, list):
            timestamps = np.array(timestamps)
        
        if isinstance(timestamps, np.ndarray):
            # Convert datetime to seconds since reference
            if isinstance(timestamps[0], datetime):
                timestamps = np.array([(ts - reference_time).total_seconds() for ts in timestamps])
        
        # Convert to tensor and normalize
        timestamps_tensor = torch.tensor(timestamps, dtype=torch.float32)
        
        # Normalize to [0, 1] range
        if timestamps_tensor.max() > timestamps_tensor.min():
            timestamps_tensor = (timestamps_tensor - timestamps_tensor.min()) / (timestamps_tensor.max() - timestamps_tensor.min())
        
        return timestamps_tensor
    
    def create_time_windows(
        self, 
        timestamps: torch.Tensor, 
        window_size: int,
        overlap: float = 0.5
    ) -> List[Tuple[float, float]]:
        """Create overlapping time windows.
        
        Args:
            timestamps: Normalized timestamps
            window_size: Size of each window (in normalized units)
            overlap: Overlap between windows (0 to 1)
            
        Returns:
            List of (start_time, end_time) tuples
        """
        step_size = window_size * (1 - overlap)
        windows = []
        
        start_time = 0.0
        while start_time < 1.0:
            end_time = min(start_time + window_size, 1.0)
            windows.append((start_time, end_time))
            start_time += step_size
        
        return windows
    
    def filter_edges_by_time(
        self, 
        edge_index: torch.Tensor,
        edge_times: torch.Tensor,
        time_window: Tuple[float, float]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Filter edges by time window.
        
        Args:
            edge_index: Edge indices [2, num_edges]
            edge_times: Edge timestamps [num_edges]
            time_window: (start_time, end_time) tuple
            
        Returns:
            Filtered edge_index and edge_times
        """
        mask = (edge_times >= time_window[0]) & (edge_times <= time_window[1])
        
        filtered_edge_index = edge_index[:, mask]
        filtered_edge_times = edge_times[mask]
        
        return filtered_edge_index, filtered_edge_times
    
    def compute_temporal_features(
        self, 
        timestamps: torch.Tensor,
        features: torch.Tensor,
        window_size: int = 10
    ) -> torch.Tensor:
        """Compute temporal features for nodes.
        
        Args:
            timestamps: Node timestamps [num_nodes]
            features: Node features [num_nodes, feature_dim]
            window_size: Window size for temporal aggregation
            
        Returns:
            Temporal features [num_nodes, temporal_feature_dim]
        """
        num_nodes = timestamps.size(0)
        feature_dim = features.size(1)
        
        # Sort nodes by timestamp
        sorted_indices = torch.argsort(timestamps)
        sorted_timestamps = timestamps[sorted_indices]
        sorted_features = features[sorted_indices]
        
        temporal_features = []
        
        for i in range(num_nodes):
            # Find nodes in temporal window
            current_time = sorted_timestamps[i]
            window_start = max(0, i - window_size // 2)
            window_end = min(num_nodes, i + window_size // 2 + 1)
            
            # Compute temporal statistics
            window_features = sorted_features[window_start:window_end]
            window_timestamps = sorted_timestamps[window_start:window_end]
            
            # Temporal statistics
            mean_features = window_features.mean(dim=0)
            std_features = window_features.std(dim=0)
            
            # Time-based features
            time_diff = current_time - window_timestamps
            time_weight = torch.exp(-time_diff / 0.1)  # Exponential decay
            
            weighted_features = (window_features * time_weight.unsqueeze(-1)).sum(dim=0) / time_weight.sum()
            
            # Combine temporal features
            node_temporal_features = torch.cat([
                mean_features,
                std_features,
                weighted_features,
                current_time.unsqueeze(0)  # Include current time
            ])
            
            temporal_features.append(node_temporal_features)
        
        return torch.stack(temporal_features)


class TemporalGraphBuilder:
    """Build temporal graphs from transaction data."""
    
    def __init__(self, time_window_days: int = 30):
        self.time_window_days = time_window_days
        self.processor = TemporalDataProcessor()
    
    def build_temporal_graph(
        self,
        transactions: List[Dict],
        node_features: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """Build temporal graph from transaction data.
        
        Args:
            transactions: List of transaction dictionaries
            node_features: Optional pre-computed node features
            
        Returns:
            Dictionary with graph components
        """
        # Extract unique nodes
        nodes = set()
        for tx in transactions:
            nodes.add(tx['source_account'])
            nodes.add(tx['target_account'])
        
        node_list = list(nodes)
        node_to_idx = {node: i for i, node in enumerate(node_list)}
        
        # Build edge information
        edges = []
        edge_times = []
        edge_weights = []
        edge_features = []
        
        for tx in transactions:
            source_idx = node_to_idx[tx['source_account']]
            target_idx = node_to_idx[tx['target_account']]
            
            edges.append([source_idx, target_idx])
            edge_times.append(tx['timestamp'])
            edge_weights.append(tx.get('amount', 1.0))
            
            # Edge features (amount, operation type, etc.)
            edge_feature = torch.tensor([
                tx.get('amount', 0.0),
                hash(tx.get('operation_type', 'payment')) % 1000 / 1000.0  # Normalized hash
            ], dtype=torch.float32)
            edge_features.append(edge_feature)
        
        # Convert to tensors
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_times = torch.tensor(edge_times, dtype=torch.float32)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        edge_features = torch.stack(edge_features)
        
        # Node timestamps (last activity time)
        node_timestamps = torch.zeros(len(node_list))
        for tx in transactions:
            source_idx = node_to_idx[tx['source_account']]
            target_idx = node_to_idx[tx['target_account']]
            tx_time = tx['timestamp']
            
            node_timestamps[source_idx] = max(node_timestamps[source_idx], tx_time)
            node_timestamps[target_idx] = max(node_timestamps[target_idx], tx_time)
        
        # Normalize timestamps
        normalized_node_times = self.processor.normalize_timestamps(node_timestamps)
        normalized_edge_times = self.processor.normalize_timestamps(edge_times)
        
        # Node features (use provided or create basic features)
        if node_features is None:
            node_features = self._create_basic_node_features(node_list, transactions)
        
        # Combine with temporal features
        temporal_node_features = self.processor.compute_temporal_features(
            normalized_node_times, node_features
        )
        
        return {
            'edge_index': edge_index,
            'edge_times': normalized_edge_times,
            'edge_weights': edge_weights,
            'edge_features': edge_features,
            'node_features': temporal_node_features,
            'node_times': normalized_node_times,
            'num_nodes': len(node_list),
            'node_mapping': node_to_idx
        }
    
    def _create_basic_node_features(
        self, 
        nodes: List[str], 
        transactions: List[Dict]
    ) -> torch.Tensor:
        """Create basic node features from transaction data."""
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        num_nodes = len(nodes)
        
        # Initialize features
        features = torch.zeros(num_nodes, 4)  # [degree, total_sent, total_received, balance]
        
        for tx in transactions:
            source_idx = node_to_idx[tx['source_account']]
            target_idx = node_to_idx[tx['target_account']]
            amount = tx.get('amount', 0.0)
            
            # Update degree
            features[source_idx, 0] += 1  # out-degree
            features[target_idx, 0] += 1  # in-degree
            
            # Update transaction amounts
            features[source_idx, 1] += amount  # total sent
            features[target_idx, 2] += amount  # total received
        
        # Compute balance (received - sent)
        features[:, 3] = features[:, 2] - features[:, 1]
        
        return features


class TemporalFeatureExtractor:
    """Extract temporal features from graph data."""
    
    def __init__(self):
        self.processor = TemporalDataProcessor()
    
    def extract_temporal_patterns(
        self,
        graph_data: Dict[str, torch.Tensor],
        pattern_types: List[str] = ['periodicity', 'trend', 'volatility']
    ) -> Dict[str, torch.Tensor]:
        """Extract various temporal patterns from graph data.
        
        Args:
            graph_data: Graph data dictionary
            pattern_types: Types of patterns to extract
            
        Returns:
            Dictionary of temporal features
        """
        features = {}
        node_times = graph_data['node_times']
        edge_times = graph_data['edge_times']
        
        if 'periodicity' in pattern_types:
            features['periodicity'] = self._extract_periodicity(node_times)
        
        if 'trend' in pattern_types:
            features['trend'] = self._extract_trend(node_times)
        
        if 'volatility' in pattern_types:
            features['volatility'] = self._extract_volatility(edge_times)
        
        return features
    
    def _extract_periodicity(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Extract periodicity features using FFT."""
        if len(timestamps) < 10:
            return torch.zeros(5)  # Return zeros if insufficient data
        
        # Compute FFT
        fft_result = torch.fft.fft(timestamps)
        power_spectrum = torch.abs(fft_result) ** 2
        
        # Get top frequencies
        top_freqs = torch.topk(power_spectrum, min(5, len(power_spectrum) // 2))[0]
        
        return top_freqs
    
    def _extract_trend(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Extract trend features."""
        if len(timestamps) < 2:
            return torch.zeros(3)
        
        # Sort timestamps
        sorted_times = torch.sort(timestamps)[0]
        
        # Linear trend
        x = torch.arange(len(sorted_times), dtype=torch.float32)
        y = sorted_times
        
        # Compute linear regression
        x_mean = x.mean()
        y_mean = y.mean()
        
        slope = torch.sum((x - x_mean) * (y - y_mean)) / torch.sum((x - x_mean) ** 2)
        intercept = y_mean - slope * x_mean
        
        # Compute R-squared
        y_pred = slope * x + intercept
        ss_res = torch.sum((y - y_pred) ** 2)
        ss_tot = torch.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return torch.tensor([slope, intercept, r_squared])
    
    def _extract_volatility(self, timestamps: torch.Tensor) -> torch.Tensor:
        """Extract volatility features."""
        if len(timestamps) < 2:
            return torch.zeros(3)
        
        # Sort timestamps
        sorted_times = torch.sort(timestamps)[0]
        
        # Compute differences
        diffs = torch.diff(sorted_times)
        
        # Volatility metrics
        mean_diff = diffs.mean()
        std_diff = diffs.std()
        max_diff = diffs.max()
        
        return torch.tensor([mean_diff, std_diff, max_diff])


class TemporalAugmentation:
    """Data augmentation for temporal graphs."""
    
    def __init__(self):
        self.processor = TemporalDataProcessor()
    
    def temporal_noise_augmentation(
        self,
        graph_data: Dict[str, torch.Tensor],
        noise_std: float = 0.01
    ) -> Dict[str, torch.Tensor]:
        """Add temporal noise to timestamps."""
        augmented_data = graph_data.copy()
        
        # Add noise to node timestamps
        node_noise = torch.randn_like(graph_data['node_times']) * noise_std
        augmented_data['node_times'] = torch.clamp(
            graph_data['node_times'] + node_noise, 0.0, 1.0
        )
        
        # Add noise to edge timestamps
        edge_noise = torch.randn_like(graph_data['edge_times']) * noise_std
        augmented_data['edge_times'] = torch.clamp(
            graph_data['edge_times'] + edge_noise, 0.0, 1.0
        )
        
        return augmented_data
    
    def temporal_dropout(
        self,
        graph_data: Dict[str, torch.Tensor],
        dropout_rate: float = 0.1
    ) -> Dict[str, torch.Tensor]:
        """Randomly drop edges based on temporal information."""
        augmented_data = graph_data.copy()
        
        # Create dropout mask based on edge times
        edge_times = graph_data['edge_times']
        
        # Higher dropout for older edges
        dropout_prob = dropout_rate * (1.0 - edge_times)  # Older edges have higher dropout
        
        mask = torch.rand_like(edge_times) > dropout_prob
        
        # Apply mask
        augmented_data['edge_index'] = graph_data['edge_index'][:, mask]
        augmented_data['edge_times'] = graph_data['edge_times'][mask]
        
        if 'edge_weights' in graph_data:
            augmented_data['edge_weights'] = graph_data['edge_weights'][mask]
        
        if 'edge_features' in graph_data:
            augmented_data['edge_features'] = graph_data['edge_features'][mask]
        
        return augmented_data
    
    def temporal_scaling(
        self,
        graph_data: Dict[str, torch.Tensor],
        scale_range: Tuple[float, float] = (0.8, 1.2)
    ) -> Dict[str, torch.Tensor]:
        """Scale temporal information."""
        augmented_data = graph_data.copy()
        
        # Random scaling factor
        scale_factor = torch.rand(1).item() * (scale_range[1] - scale_range[0]) + scale_range[0]
        
        # Scale timestamps
        augmented_data['node_times'] = graph_data['node_times'] * scale_factor
        augmented_data['edge_times'] = graph_data['edge_times'] * scale_factor
        
        # Clamp to valid range
        augmented_data['node_times'] = torch.clamp(augmented_data['node_times'], 0.0, 1.0)
        augmented_data['edge_times'] = torch.clamp(augmented_data['edge_times'], 0.0, 1.0)
        
        return augmented_data


class TemporalMetrics:
    """Metrics for evaluating temporal models."""
    
    @staticmethod
    def temporal_consistency_loss(
        predictions: torch.Tensor,
        timestamps: torch.Tensor,
        consistency_weight: float = 0.1
    ) -> torch.Tensor:
        """Compute temporal consistency loss."""
        # Sort by timestamps
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        
        # Compute temporal consistency (smoothness)
        temporal_diff = torch.diff(sorted_predictions, dim=0)
        consistency_loss = torch.mean(temporal_diff ** 2)
        
        return consistency_weight * consistency_loss
    
    @staticmethod
    def temporal_auc(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timestamps: torch.Tensor,
        time_window: float = 0.1
    ) -> float:
        """Compute time-windowed AUC."""
        # Create time windows
        windows = []
        start_time = 0.0
        while start_time < 1.0:
            end_time = min(start_time + time_window, 1.0)
            windows.append((start_time, end_time))
            start_time += time_window
        
        aucs = []
        
        for window_start, window_end in windows:
            # Filter data by time window
            mask = (timestamps >= window_start) & (timestamps <= window_end)
            
            if mask.sum() > 1:
                window_preds = predictions[mask]
                window_targets = targets[mask]
                
                # Compute AUC for this window
                try:
                    from sklearn.metrics import roc_auc_score  # noqa: E402
                    auc = roc_auc_score(window_targets.cpu().numpy(), window_preds.cpu().numpy())
                    aucs.append(auc)
                except Exception:
                    pass
        
        return np.mean(aucs) if aucs else 0.0
    
    @staticmethod
    def temporal_prediction_accuracy(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        timestamps: torch.Tensor,
        tolerance: float = 0.05
    ) -> float:
        """Compute prediction accuracy with temporal tolerance."""
        # Sort by timestamps
        sorted_indices = torch.argsort(timestamps)
        sorted_predictions = predictions[sorted_indices]
        sorted_targets = targets[sorted_indices]
        
        # Compute predictions with tolerance
        predicted_classes = torch.argmax(sorted_predictions, dim=1)
        correct = (predicted_classes == sorted_targets).float()
        
        # Apply temporal smoothing
        kernel_size = max(3, int(len(correct) * tolerance))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Smooth accuracy over time
        smoothed_correct = torch.nn.functional.avg_pool1d(
            correct.unsqueeze(0).unsqueeze(0).float(),
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        ).squeeze()
        
        return smoothed_correct.mean().item()


class TemporalVisualization:
    """Utilities for visualizing temporal graph data."""
    
    @staticmethod
    def plot_temporal_distribution(
        timestamps: torch.Tensor,
        title: str = "Temporal Distribution"
    ) -> None:
        """Plot temporal distribution of data."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.hist(timestamps.cpu().numpy(), bins=50, alpha=0.7)
        plt.title(title)
        plt.xlabel("Normalized Time")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.show()
    
    @staticmethod
    def plot_temporal_heatmap(
        features: torch.Tensor,
        timestamps: torch.Tensor,
        title: str = "Temporal Feature Heatmap"
    ) -> None:
        """Plot temporal feature heatmap."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Sort by timestamps
        sorted_indices = torch.argsort(timestamps)
        sorted_features = features[sorted_indices]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(sorted_features.cpu().numpy(), cmap='viridis')
        plt.title(title)
        plt.xlabel("Feature Index")
        plt.ylabel("Node (sorted by time)")
        plt.show()
    
    @staticmethod
    def plot_temporal_graph_evolution(
        graph_data: Dict[str, torch.Tensor],
        num_snapshots: int = 5
    ) -> None:
        """Plot graph evolution over time."""
        import matplotlib.pyplot as plt
        import networkx as nx
        
        # Create time windows
        processor = TemporalDataProcessor()
        windows = processor.create_time_windows(
            graph_data['node_times'], 
            window_size=1.0 / num_snapshots,
            overlap=0.0
        )
        
        fig, axes = plt.subplots(1, num_snapshots, figsize=(20, 4))
        
        for i, (start_time, end_time) in enumerate(windows):
            # Filter edges by time window
            edge_mask = (graph_data['edge_times'] >= start_time) & (graph_data['edge_times'] <= end_time)
            filtered_edge_index = graph_data['edge_index'][:, edge_mask]
            
            # Create NetworkX graph
            G = nx.Graph()
            G.add_nodes_from(range(graph_data['num_nodes']))
            G.add_edges_from(filtered_edge_index.t().tolist())
            
            # Plot graph
            ax = axes[i]
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax, node_size=50, node_color='lightblue', 
                   edge_color='gray', with_labels=False)
            ax.set_title(f"Time Window {i+1}\n({start_time:.2f} - {end_time:.2f})")
        
        plt.tight_layout()
        plt.show()
