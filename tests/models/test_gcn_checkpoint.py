"""Checkpoint format and compatibility rules for the baseline GCN (issue #1201)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from astroml.models.checkpoint import (  # noqa: E402
    CHECKPOINT_FORMAT,
    CHECKPOINT_VERSION,
    CheckpointCompatibilityError,
    GCNConfig,
    gcn_config,
    load_gcn_checkpoint,
    save_gcn_checkpoint,
)
from astroml.models.gcn import GCN  # noqa: E402
from astroml.utils.exceptions import AstroMLError, ModelError  # noqa: E402

EDGES = torch.tensor([[0, 1, 2], [1, 0, 3]], dtype=torch.long)


class _Unsafe:
    """A module-level class, so a checkpoint holding it can be written and then refused."""


def _payload(model: GCN, **overrides: Any) -> dict[str, Any]:
    """A valid checkpoint payload, with some fields replaced, for building bad files."""
    payload: dict[str, Any] = {
        "format": CHECKPOINT_FORMAT,
        "version": CHECKPOINT_VERSION,
        "config": gcn_config(model).to_dict(),
        "state_dict": dict(model.state_dict()),
        "metadata": {},
    }
    payload.update(overrides)
    return payload


@pytest.fixture()
def model() -> GCN:
    torch.manual_seed(0)
    return GCN(4, 8, 3, dropout=0.3).eval()


class TestRoundTrip:
    def test_loaded_model_gives_the_same_output(self, model: GCN, tmp_path: Path) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt")
        loaded = load_gcn_checkpoint(path)

        x = torch.randn(5, 4)
        assert torch.allclose(model(x, EDGES), loaded.model(x, EDGES))

    def test_loaded_model_is_in_evaluation_mode(self, model: GCN, tmp_path: Path) -> None:
        model.train()
        loaded = load_gcn_checkpoint(save_gcn_checkpoint(model, tmp_path / "gcn.pt"))
        assert loaded.model.training is False

    def test_the_architecture_is_read_back_from_the_file(self, model: GCN, tmp_path: Path) -> None:
        loaded = load_gcn_checkpoint(save_gcn_checkpoint(model, tmp_path / "gcn.pt"))
        assert loaded.config == GCNConfig(4, 8, 3, dropout=0.3)
        assert loaded.version == CHECKPOINT_VERSION

    def test_metadata_records_the_environment_and_caller_values(
        self, model: GCN, tmp_path: Path
    ) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt", metadata={"run_id": "abc"})
        metadata = load_gcn_checkpoint(path).metadata

        assert metadata["run_id"] == "abc"
        assert metadata["environment"]["packages"]["torch"] == torch.__version__
        assert "python" in metadata["environment"]

    def test_the_file_loads_without_executing_pickled_code(
        self, model: GCN, tmp_path: Path
    ) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt")
        # weights_only=True refuses anything but tensors and plain values.
        assert torch.load(path, weights_only=True)["format"] == CHECKPOINT_FORMAT


class TestCompatibilityRules:
    def test_a_matching_expected_architecture_is_accepted(self, model: GCN, tmp_path: Path) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt")
        assert load_gcn_checkpoint(path, expected=GCNConfig(4, 8, 3)).model is not None

    @pytest.mark.parametrize(
        "expected",
        [GCNConfig(5, 8, 3), GCNConfig(4, 16, 3), GCNConfig(4, 8, 7)],
        ids=["input_dim", "hidden_dim", "output_dim"],
    )
    def test_a_different_architecture_is_refused(
        self, model: GCN, tmp_path: Path, expected: GCNConfig
    ) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt")
        with pytest.raises(CheckpointCompatibilityError, match="architecture mismatch"):
            load_gcn_checkpoint(path, expected=expected)

    def test_dropout_is_not_part_of_the_architecture(self, model: GCN, tmp_path: Path) -> None:
        path = save_gcn_checkpoint(model, tmp_path / "gcn.pt")
        loaded = load_gcn_checkpoint(path, expected=GCNConfig(4, 8, 3, dropout=0.9))
        assert loaded.config.dropout == 0.3

    def test_a_newer_format_version_is_refused_not_guessed_at(
        self, model: GCN, tmp_path: Path
    ) -> None:
        path = tmp_path / "future.pt"
        torch.save(_payload(model, version=CHECKPOINT_VERSION + 1), path)
        with pytest.raises(CheckpointCompatibilityError, match="newer than"):
            load_gcn_checkpoint(path)

    def test_a_version_below_the_minimum_is_refused(self, model: GCN, tmp_path: Path) -> None:
        path = tmp_path / "old.pt"
        torch.save(_payload(model, version=0), path)
        with pytest.raises(CheckpointCompatibilityError, match="older than"):
            load_gcn_checkpoint(path)

    def test_a_non_integer_version_is_refused(self, model: GCN, tmp_path: Path) -> None:
        path = tmp_path / "bad-version.pt"
        torch.save(_payload(model, version="1"), path)
        with pytest.raises(CheckpointCompatibilityError, match="version must be an integer"):
            load_gcn_checkpoint(path)

    def test_another_kind_of_file_is_refused(self, model: GCN, tmp_path: Path) -> None:
        path = tmp_path / "other.pt"
        torch.save(_payload(model, format="something.else"), path)
        with pytest.raises(CheckpointCompatibilityError, match="not an"):
            load_gcn_checkpoint(path)

    def test_a_bare_state_dict_is_refused(self, model: GCN, tmp_path: Path) -> None:
        path = tmp_path / "bare.pt"
        torch.save(model.state_dict(), path)
        with pytest.raises(CheckpointCompatibilityError, match="not an"):
            load_gcn_checkpoint(path)

    @pytest.mark.parametrize(
        "config",
        [
            None,
            {"input_dim": 4, "hidden_dim": 8},
            {"input_dim": 0, "hidden_dim": 8, "output_dim": 3},
            {"input_dim": 4.5, "hidden_dim": 8, "output_dim": 3},
            {"input_dim": True, "hidden_dim": 8, "output_dim": 3},
            {"input_dim": 4, "hidden_dim": 8, "output_dim": 3, "dropout": 1.0},
            {"input_dim": 4, "hidden_dim": 8, "output_dim": 3, "dropout": "high"},
        ],
        ids=[
            "missing",
            "missing-key",
            "zero-dim",
            "float-dim",
            "bool-dim",
            "dropout-one",
            "dropout-string",
        ],
    )
    def test_an_incomplete_or_invalid_config_is_refused(
        self, model: GCN, tmp_path: Path, config: Any
    ) -> None:
        path = tmp_path / "bad-config.pt"
        torch.save(_payload(model, config=config), path)
        with pytest.raises(CheckpointCompatibilityError, match="config"):
            load_gcn_checkpoint(path)

    def test_missing_weights_are_named(self, model: GCN, tmp_path: Path) -> None:
        state = dict(model.state_dict())
        dropped = next(iter(state))
        del state[dropped]
        path = tmp_path / "missing.pt"
        torch.save(_payload(model, state_dict=state), path)

        with pytest.raises(CheckpointCompatibilityError, match=f"missing keys.*{dropped}"):
            load_gcn_checkpoint(path)

    def test_unexpected_weights_are_named(self, model: GCN, tmp_path: Path) -> None:
        state = dict(model.state_dict())
        state["conv3.weight"] = torch.zeros(1)
        path = tmp_path / "extra.pt"
        torch.save(_payload(model, state_dict=state), path)

        with pytest.raises(CheckpointCompatibilityError, match="unexpected keys.*conv3.weight"):
            load_gcn_checkpoint(path)

    def test_weights_of_the_wrong_shape_are_named(self, model: GCN, tmp_path: Path) -> None:
        state = dict(model.state_dict())
        victim = "conv1.bias"
        state[victim] = torch.zeros(99)
        path = tmp_path / "shape.pt"
        torch.save(_payload(model, state_dict=state), path)

        with pytest.raises(CheckpointCompatibilityError, match=f"wrong shape.*{victim}"):
            load_gcn_checkpoint(path)

    def test_a_missing_state_dict_is_refused(self, model: GCN, tmp_path: Path) -> None:
        path = tmp_path / "no-state.pt"
        torch.save(_payload(model, state_dict=None), path)
        with pytest.raises(CheckpointCompatibilityError, match="state_dict"):
            load_gcn_checkpoint(path)


class TestUnreadableFiles:
    def test_a_missing_file_is_a_file_not_found_error(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_gcn_checkpoint(tmp_path / "absent.pt")

    def test_a_corrupt_file_is_a_compatibility_error(self, tmp_path: Path) -> None:
        path = tmp_path / "corrupt.pt"
        path.write_bytes(b"this is not a torch file")
        with pytest.raises(CheckpointCompatibilityError, match="could not be read"):
            load_gcn_checkpoint(path)

    def test_a_file_with_pickled_code_is_refused_not_executed(
        self, model: GCN, tmp_path: Path
    ) -> None:
        path = tmp_path / "unsafe.pt"
        torch.save(_payload(model, metadata={"object": _Unsafe()}), path)
        with pytest.raises(CheckpointCompatibilityError, match="could not be read"):
            load_gcn_checkpoint(path)


class TestErrorContract:
    def test_it_is_a_model_error_with_a_stable_code(self, tmp_path: Path) -> None:
        path = tmp_path / "other.pt"
        torch.save({"format": "x"}, path)

        with pytest.raises(CheckpointCompatibilityError) as raised:
            load_gcn_checkpoint(path)

        error = raised.value
        assert isinstance(error, ModelError) and isinstance(error, AstroMLError)
        assert error.code == "CHECKPOINT_INCOMPATIBLE"
        assert error.retryable is False

    def test_the_public_form_does_not_leak_the_path_or_reason(self, tmp_path: Path) -> None:
        path = tmp_path / "secret-location.pt"
        torch.save({"format": "x"}, path)

        with pytest.raises(CheckpointCompatibilityError) as raised:
            load_gcn_checkpoint(path)

        public = json.dumps(raised.value.to_public_dict())
        assert "secret-location" not in public
        assert "CHECKPOINT_INCOMPATIBLE" in public


class TestSaving:
    def test_a_failed_write_leaves_the_previous_checkpoint_intact(
        self, model: GCN, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "gcn.pt"
        save_gcn_checkpoint(model, path, metadata={"generation": 1})
        good_bytes = path.read_bytes()

        def interrupted(obj: Any, target: Any, *args: Any, **kwargs: Any) -> None:
            Path(target).write_bytes(b"half a checkpoint")
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", interrupted)
        with pytest.raises(OSError, match="disk full"):
            save_gcn_checkpoint(model, path, metadata={"generation": 2})

        assert path.read_bytes() == good_bytes
        assert load_gcn_checkpoint(path).metadata["generation"] == 1
        assert not Path(str(path) + ".tmp").exists()

    def test_metadata_that_is_not_json_is_rejected_before_anything_is_written(
        self, model: GCN, tmp_path: Path
    ) -> None:
        path = tmp_path / "gcn.pt"
        with pytest.raises(ValueError, match="JSON-serialisable"):
            save_gcn_checkpoint(model, path, metadata={"bad": object()})
        assert not path.exists()
