from astroml.training.explainability.reports import ReportGenerator
from astroml.training.explainability.visualizations import plot_shap
def test_reports() -> None:
    r = ReportGenerator()
    assert r.generate() == "report"
    plot_shap()
