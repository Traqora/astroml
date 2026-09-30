from astroml.api.routers.explainability import get_explainability


def test_get_explain() -> None:
    assert get_explainability()["status"] == "ok"
