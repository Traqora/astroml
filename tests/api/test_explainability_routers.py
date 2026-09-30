from astroml.api.routers.explainability_reports import get_reports
def test_get_reports() -> None:
    assert get_reports()["status"] == "ok"
