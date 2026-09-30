from astroml.observability.model_monitor import ModelMonitor
from astroml.observability.metrics_collector import MetricsCollector
from astroml.observability.drift_detector import DriftDetector
def test_observability() -> None:
    m = ModelMonitor()
    m.monitor()
    c = MetricsCollector()
    c.collect()
    d = DriftDetector()
    assert d.detect() is False
