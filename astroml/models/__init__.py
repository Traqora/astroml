"""Machine learning models for AstroML."""

from .gcn import GCN
from .graph_sage import AGGREGATIONS, GraphSAGE, SAGEAggregator
from .link_prediction import GCNEncoder, LinkPredictor
from .registry import ModelRegistry
from .rollback import RollbackGuard
from .sage_encoder import InductiveSAGEEncoder
from .temporal import (
    TemporalAttention,
    TemporalEdgeConv,
    TemporalEncoding,
    TemporalGAT,
    TemporalGCN,
    TemporalGraphSAGE,
    TemporalGraphTransformer,
    TemporalModelFactory,
)
from .tgn import (
    MemoryState,
    TemporalGraphNetwork,
    TimeEncoder,
)

try:
    from .deep_svdd import DeepSVDD, DeepSVDDNetwork
    from .deep_svdd_trainer import DeepSVDDTrainer, FraudDetectionDeepSVDD
except ImportError:
    pass

__all__ = [
    'GCN',
    'GraphSAGE',
    'SAGEAggregator',
    'AGGREGATIONS',
    'TemporalGCN',
    'TemporalGraphSAGE',
    'TemporalGAT',
    'TemporalGraphTransformer',
    'TemporalEdgeConv',
    'TemporalEncoding',
    'TemporalAttention',
    'TemporalModelFactory',
    'TemporalGraphNetwork',
    'TimeEncoder',
    'MemoryState',
    'DeepSVDD',
    'DeepSVDDNetwork',
    'DeepSVDDTrainer',
    'FraudDetectionDeepSVDD',
    'InductiveSAGEEncoder',
    'GCNEncoder',
    'LinkPredictor',
]
