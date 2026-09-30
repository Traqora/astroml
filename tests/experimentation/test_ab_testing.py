from astroml.experimentation.ab_testing import ABTester
from astroml.experimentation.traffic_splitter import TrafficSplitter
from astroml.experimentation.metrics_collector import MetricsCollector
from astroml.api.routers.experiments import ExperimentsRouter
from astroml.tracking.experiment_tracker import ExperimentTracker

def test_traffic_splitter():
    splitter = TrafficSplitter(default_split=1.0)
    assert splitter.split_random("user1") == "treatment"
    assert splitter.split_cookie("cookie1") in ["control", "treatment"]

def test_metrics_collector():
    collector = MetricsCollector()
    collector.collect("control", 1.0)
    assert len(collector.get_metrics()["control"]) == 1

def test_ab_tester():
    tester = ABTester()
    group = tester.assign_group("user1")
    assert group in ["control", "treatment"]
    tester.record_metric("treatment", 1.0)
    tester.record_metric("control", 0.0)
    sig, p = tester.check_significance()
    assert sig is True

def test_experiments_router():
    router = ExperimentsRouter()
    router.create_experiment("exp1")
    assert router.get_experiment("exp1")["status"] == "running"

def test_experiment_tracker():
    tracker = ExperimentTracker()
    tracker.log_experiment("exp1", "winner")
    assert tracker.get_experiment_log("exp1") == "winner"
