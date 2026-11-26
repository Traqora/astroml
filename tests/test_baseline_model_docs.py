"""Keeps docs/baseline-model-assumptions.md in step with the baseline code (#1203).

The baseline is the 2-layer GCN in ``astroml/models/gcn.py`` trained by
``astroml/training/train_gcn.py``. The documented hyper-parameters are read
back out of that source with ``ast`` (no torch import, so this runs without
optional dependencies) and compared with what the document states. If someone
changes the model or training defaults without updating the document, or the
reverse, this fails.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "baseline-model-assumptions.md"
GCN_SRC = ROOT / "astroml" / "models" / "gcn.py"
TRAIN_SRC = ROOT / "astroml" / "training" / "train_gcn.py"

REQUIRED_SECTIONS = [
    "## Model",
    "## Training protocol",
    "## Deterministic setup (issue #1206)",
    "## Early stopping (issue #1208)",
    "## Input contract",
    "## Output contract",
    "## Supported boundaries",
    "## Not guaranteed",
]


def _call_keywords(tree: ast.AST, func_name: str) -> dict[str, object]:
    """Literal keyword arguments of the first call to ``func_name`` (dotted or bare)."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else getattr(node.func, "id", "")
            )
            if name == func_name:
                return {
                    kw.arg: ast.literal_eval(kw.value)
                    for kw in node.keywords
                    if kw.arg and isinstance(kw.value, (ast.Constant, ast.UnaryOp))
                }
    raise AssertionError(f"no call to {func_name} found")


def code_baseline() -> dict[str, object]:
    """Baseline settings as they exist in the source."""
    train_tree = ast.parse(TRAIN_SRC.read_text(encoding="utf-8"))
    gcn_tree = ast.parse(GCN_SRC.read_text(encoding="utf-8"))

    model_kwargs = _call_keywords(train_tree, "GCN")
    adam_kwargs = _call_keywords(train_tree, "Adam")

    train_fn = next(
        n for n in ast.walk(train_tree) if isinstance(n, ast.FunctionDef) and n.name == "train"
    )
    defaults = {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(
            train_fn.args.args[-len(train_fn.args.defaults) :], train_fn.args.defaults
        )
    }

    gcn_cls = next(n for n in ast.walk(gcn_tree) if isinstance(n, ast.ClassDef) and n.name == "GCN")
    init = next(n for n in gcn_cls.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
    init_defaults = {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(init.args.args[-len(init.args.defaults) :], init.args.defaults)
    }
    conv_layers = sum(
        1
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "GCNConv"
    )

    return {
        "hidden_dim": model_kwargs["hidden_dim"],
        "dropout": model_kwargs["dropout"],
        "model_default_dropout": init_defaults["dropout"],
        "lr": adam_kwargs["lr"],
        "weight_decay": adam_kwargs["weight_decay"],
        "epochs": defaults["epochs"],
        "eval_every": defaults["eval_every"],
        "seed": defaults["seed"],
        "layers": conv_layers,
    }


def documented_baseline(text: str) -> dict[str, object]:
    """Pull the same settings out of the document's tables."""

    def cell(label: str) -> str:
        match = re.search(rf"^\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|", text, re.MULTILINE)
        assert match, f"document has no table row for {label!r}"
        return match.group(1)

    adam = cell("Optimiser")
    return {
        "layers": int(cell("Layers")),
        "hidden_dim": int(cell("`hidden_dim`")),
        "dropout": float(cell("`dropout`")),
        "lr": float(re.search(r"learning rate `([^`]+)`", adam).group(1)),
        "weight_decay": float(re.search(r"weight decay `([^`]+)`", adam).group(1)),
        "epochs": int(re.match(r"\d+", cell("Epochs")).group(0)),
        "eval_every": int(re.search(r"every (\d+) epochs", cell("Validation cadence")).group(1)),
    }


def _doc_text() -> str:
    assert DOC.is_file(), "docs/baseline-model-assumptions.md is missing"
    return DOC.read_text(encoding="utf-8")


def test_document_has_every_required_section() -> None:
    text = _doc_text()
    missing = [s for s in REQUIRED_SECTIONS if s not in text]
    assert not missing, f"baseline doc is missing sections: {missing}"


def test_documented_hyperparameters_match_the_code() -> None:
    code = code_baseline()
    documented = documented_baseline(_doc_text())

    for key, doc_value in documented.items():
        assert (
            code[key] == doc_value
        ), f"{key}: document says {doc_value!r} but the code uses {code[key]!r}"
    # The model's own default and the value the training loop passes agree.
    assert code["model_default_dropout"] == code["dropout"]


def test_document_states_the_input_contract_the_code_enforces() -> None:
    text = _doc_text()
    for needle in ("[2, E]", "[0, N)", "disjoint", "validate_edge_index", "validate_masks"):
        assert needle in text, f"input contract no longer mentions {needle!r}"

    # ...and those validators really are called by train().
    source = TRAIN_SRC.read_text(encoding="utf-8")
    assert "validate_edge_index(" in source and "validate_masks(" in source
    assert "set_seed(seed)" in source
    assert code_baseline()["seed"] == 42


def test_document_calls_out_what_is_not_supported() -> None:
    text = _doc_text()
    for needle in (
        "Transductive",
        "Inductive",
        "seed=42",
        "PYTHONHASHSEED",
        "val_loss",
        "patience",
        "Bit-for-bit",
    ):
        assert needle.lower() in text.lower(), f"boundary {needle!r} is not documented"


def test_a_stale_document_value_is_detected() -> None:
    """The comparison is meaningful: change a documented number and it must fail."""
    stale = _doc_text().replace("| `hidden_dim`    | 16 ", "| `hidden_dim`    | 64 ")
    assert stale != _doc_text(), "test fixture no longer matches the document table"

    with pytest.raises(AssertionError, match="hidden_dim"):
        code = code_baseline()
        for key, doc_value in documented_baseline(stale).items():
            assert code[key] == doc_value, f"{key}: document says {doc_value!r}"


def test_older_docs_that_disagree_are_flagged() -> None:
    """The page must warn about the config/tutorial that do not describe the baseline."""
    text = _doc_text()
    assert "configs/model/gcn.yaml" in text
    assert "docs/tutorial-gcn.md" in text
    # The claim is true: gcn.py has no train_gcn.
    assert "def train_gcn" not in GCN_SRC.read_text(encoding="utf-8")
