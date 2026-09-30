from astroml.api.routers.synthetic_data import get_synthetic
def test_get_synthetic() -> None:
    assert get_synthetic()["status"] == "ok"
