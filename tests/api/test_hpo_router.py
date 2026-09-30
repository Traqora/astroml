from astroml.api.routers.hpo import get_hpo


def test_get_hpo() -> None:
    assert get_hpo()["status"] == "ok"
