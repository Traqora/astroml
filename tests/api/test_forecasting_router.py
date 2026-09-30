from astroml.api.routers.forecasting import get_forecast


def test_get_forecast() -> None:
    assert get_forecast()["status"] == "ok"
