"""Machine learning models for AstroML."""

from .gcn import GCN
from .temporal import (
    TemporalGCN,
    TemporalGraphSAGE,
    TemporalGAT,
    TemporalGraphTransformer,
    TemporalEdgeConv,
    TemporalEncoding,
    TemporalAttention,
    TemporalModelFactory,
)
from .sage_encoder import InductiveSAGEEncoder
from .link_prediction import LinkPredictor, GCNEncoder

try:
    from .deep_svdd import DeepSVDD, DeepSVDDNetwork
    from .deep_svdd_trainer import DeepSVDDTrainer, FraudDetectionDeepSVDD
except ImportError:
    pass

__all__ = [
    'GCN',
    'TemporalGCN',
    'TemporalGraphSAGE',
    'TemporalGAT',
    'TemporalGraphTransformer',
    'TemporalEdgeConv',
    'TemporalEncoding',
    'TemporalAttention',
    'TemporalModelFactory',
    'DeepSVDD',
    'DeepSVDDNetwork',
    'DeepSVDDTrainer',
    'FraudDetectionDeepSVDD',
    'InductiveSAGEEncoder',
    'GCNEncoder',
    'LinkPredictor',
]
