import os
import sys
from importlib import reload


def test_parse_command_line_seed_sets_astroml_seed(monkeypatch):
    """Ensure the top-level --seed CLI flag is parsed and preserved for Hydra."""
    monkeypatch.delenv("ASTROML_SEED", raising=False)
    monkeypatch.setattr(sys, "argv", ["train.py", "--seed", "123", "experiment=debug"])

    import train
    reload(train)

    train._parse_command_line_seed()

    assert os.environ["ASTROML_SEED"] == "123"
    assert sys.argv == ["train.py", "experiment=debug"]
