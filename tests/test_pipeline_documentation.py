"""Tests for automated ML pipeline documentation generation.

Covers #646.
"""

from __future__ import annotations

import json

import pytest

from astroml.pipeline.documentation.data_flow import (
    DataFlowDiagram,
    DataFlowNode,
    NodeKind,
)
from astroml.pipeline.documentation.generator import (
    IOSpec,
    PipelineDocGenerator,
    PipelineMetadata,
    StageMetadata,
    extract_metadata,
)
from astroml.pipeline.documentation.model_card import (
    MetricEntry,
    ModelCard,
    ModelCardBuilder,
    ModelDetails,
)
from astroml.pipeline.documentation.templates import available_templates, load_template


@pytest.fixture
def metadata() -> PipelineMetadata:
    """A complete, valid pipeline definition."""
    return PipelineMetadata(
        name="fraud-scoring",
        description="Nightly anomaly scoring for Stellar accounts.",
        version="1.2.0",
        owner="ml-platform@example.com",
        framework="pytorch",
        schedule="0 2 * * *",
        stages=[
            StageMetadata("ingest", NodeKind.SOURCE, description="Read the ledger."),
            StageMetadata("features", NodeKind.FEATURE, depends_on=["ingest"]),
            StageMetadata("score", NodeKind.MODEL, depends_on=["features"]),
            StageMetadata("publish", NodeKind.SINK, depends_on=["score"]),
        ],
        inputs=[IOSpec("ledger", location="s3://ledger")],
        outputs=[IOSpec("alerts", location="postgres://alerts")],
        parameters={"threshold": 0.9},
        dependencies=["torch>=2.0"],
    )


# ─── DataFlowDiagram ─────────────────────────────────────────────────────────


class TestDataFlowDiagram:
    """Graph construction, validation and rendering."""

    def test_nodes_and_edges_round_trip(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("a", "A", NodeKind.SOURCE))
        diagram.add_node(DataFlowNode("b", "B", NodeKind.SINK))
        diagram.add_edge("a", "b", label="rows")

        restored = DataFlowDiagram.from_dict(diagram.to_dict())
        assert [n.node_id for n in restored.nodes] == ["a", "b"]
        assert restored.edges[0].label == "rows"

    def test_empty_node_id_rejected(self) -> None:
        with pytest.raises(ValueError):
            DataFlowNode("", "label")

    def test_edge_to_unknown_node_rejected(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("a", "A"))
        with pytest.raises(KeyError):
            diagram.add_edge("a", "missing")

    def test_topological_order_follows_dependencies(self) -> None:
        diagram = DataFlowDiagram.from_stages(
            "p",
            [("read", NodeKind.SOURCE), ("map", NodeKind.TRANSFORM), ("write", NodeKind.SINK)],
        )
        assert diagram.topological_order() == ["s0", "s1", "s2"]

    def test_cycle_is_detected(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("a", "A"))
        diagram.add_node(DataFlowNode("b", "B"))
        diagram.add_edge("a", "b")
        diagram.add_edge("b", "a")
        with pytest.raises(ValueError, match="cycle"):
            diagram.topological_order()
        assert any("cycle" in problem for problem in diagram.validate())

    def test_validate_reports_missing_source_and_sink(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("a", "A", NodeKind.TRANSFORM))
        diagram.add_node(DataFlowNode("b", "B", NodeKind.TRANSFORM))
        diagram.add_edge("a", "b")
        problems = diagram.validate()
        assert "pipeline has no source node" in problems
        assert "pipeline has no sink node" in problems

    def test_validate_reports_orphans(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("a", "A", NodeKind.SOURCE))
        diagram.add_node(DataFlowNode("b", "B", NodeKind.SINK))
        assert any("not connected" in problem for problem in diagram.validate())

    def test_empty_diagram_reports_no_nodes(self) -> None:
        assert DataFlowDiagram("p").validate() == ["diagram has no nodes"]

    def test_mermaid_rendering(self) -> None:
        diagram = DataFlowDiagram("p")
        diagram.add_node(DataFlowNode("read data", 'Read "ledger"', NodeKind.SOURCE))
        diagram.add_node(DataFlowNode("model", "Score", NodeKind.MODEL))
        diagram.add_edge("read data", "model", label="features")

        mermaid = diagram.to_mermaid()
        assert mermaid.startswith("flowchart LR")
        # Spaces become underscores and inner quotes become apostrophes so the
        # label delimiters stay unambiguous.
        assert "read_data" in mermaid
        assert "\"Read 'ledger'\"" in mermaid
        assert '-- "features" -->' in mermaid

    def test_mermaid_direction_validated(self) -> None:
        with pytest.raises(ValueError):
            DataFlowDiagram("p").to_mermaid(direction="DIAGONAL")

    def test_dot_rendering_escapes_quotes(self) -> None:
        diagram = DataFlowDiagram('p"q')
        diagram.add_node(DataFlowNode("a", 'A "quoted"', NodeKind.SOURCE))
        dot = diagram.to_dot()
        assert dot.startswith('digraph "p\\"q"')
        assert 'A \\"quoted\\"' in dot

    def test_json_rendering(self) -> None:
        diagram = DataFlowDiagram.from_stages("p", [("a", NodeKind.SOURCE)])
        assert json.loads(diagram.to_json())["nodes"][0]["kind"] == "source"


# ─── ModelCard ───────────────────────────────────────────────────────────────


class TestModelCard:
    """Google Model Cards support."""

    def test_builder_populates_sections(self) -> None:
        card = (
            ModelCardBuilder("fraud-gnn", version="2.0.0")
            .with_overview("GraphSAGE anomaly scorer.")
            .with_owners(["ml-platform@example.com"])
            .with_model_type("GraphSAGE", framework="pytorch")
            .with_licenses(["Apache-2.0"])
            .with_references(["https://example.com/paper"])
            .with_intended_use(
                primary_uses=["Flag suspicious accounts"],
                primary_users=["Fraud analysts"],
                out_of_scope_uses=["Automated fund freezing"],
            )
            .with_factors(["account age"])
            .with_metric("roc_auc", 0.94, threshold=0.9)
            .with_metric("demographic_parity", 0.02, fairness=True)
            .with_training_data("Ledger snapshot", sources=["s3://ledger"], size=1_000)
            .with_evaluation_data("Held-out month", size=200, split_strategy="time-based")
            .with_limitations(["Untested on testnet data"])
            .with_caveats(["Recalibrate quarterly"])
            .with_ethical_considerations(["Scores must not auto-freeze funds."])
            .build()
        )

        assert card.is_valid()
        assert card.validate() == []
        assert card.model_details.version == "2.0.0"
        assert card.quantitative_analysis.fairness_metrics[0].name == "demographic_parity"

    def test_builder_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError):
            ModelCardBuilder("")

    def test_with_metrics_bulk_add(self) -> None:
        card = ModelCardBuilder("m").with_metrics({"a": 1.0, "b": 2.0}).build()
        assert len(card.quantitative_analysis.performance_metrics) == 2

    def test_validate_lists_missing_sections(self) -> None:
        card = ModelCard(model_details=ModelDetails(name="m"))
        problems = card.validate()
        assert not card.is_valid()
        assert any("overview" in problem for problem in problems)
        assert any("ethical_considerations" in problem for problem in problems)

    def test_metric_threshold_evaluation(self) -> None:
        assert MetricEntry("auc", 0.95, threshold=0.9).passes() is True
        assert MetricEntry("auc", 0.85, threshold=0.9).passes() is False
        assert MetricEntry("auc", 0.85).passes() is None
        assert MetricEntry("fpr", 0.02, threshold=0.05, higher_is_better=False).passes() is True

    def test_markdown_includes_metric_table(self) -> None:
        card = ModelCardBuilder("fraud-gnn").with_metric("roc_auc", 0.94, threshold=0.9).build()
        markdown = card.to_markdown()
        assert "# Model Card — fraud-gnn" in markdown
        assert "| roc_auc | overall | 0.94 | 0.9 | pass |" in markdown
        assert "_None documented._" in markdown

    def test_json_round_trip(self) -> None:
        card = ModelCardBuilder("m").with_overview("o").build()
        assert json.loads(card.to_json())["model_details"]["overview"] == "o"

    def test_write_produces_both_formats(self, tmp_path) -> None:
        card = ModelCardBuilder("Fraud GNN!").build()
        written = card.write(tmp_path)
        assert {path.suffix for path in written} == {".md", ".json"}
        assert all(path.is_file() for path in written)
        assert written[0].name.startswith("fraud_gnn")

    def test_write_rejects_unknown_format(self, tmp_path) -> None:
        with pytest.raises(ValueError):
            ModelCardBuilder("m").build().write(tmp_path, formats=("pdf",))


# ─── Templates ───────────────────────────────────────────────────────────────


class TestTemplates:
    """Packaged template loading."""

    def test_expected_templates_are_packaged(self) -> None:
        assert "pipeline.md.tmpl" in available_templates()
        assert "model_card.md.tmpl" in available_templates()

    def test_load_template_returns_content(self) -> None:
        assert "$name" in load_template("pipeline.md.tmpl")

    def test_missing_template_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_template("nope.tmpl")

    @pytest.mark.parametrize("name", ["../secrets.tmpl", "a/b.tmpl", ".hidden"])
    def test_traversal_names_rejected(self, name: str) -> None:
        with pytest.raises(ValueError):
            load_template(name)


# ─── PipelineMetadata & generator ────────────────────────────────────────────


class TestPipelineMetadata:
    """Metadata validation and diagram derivation."""

    def test_empty_name_rejected(self) -> None:
        with pytest.raises(ValueError):
            PipelineMetadata(name="")

    def test_complete_metadata_validates(self, metadata: PipelineMetadata) -> None:
        assert metadata.validate() == []

    def test_incomplete_metadata_reports_problems(self) -> None:
        problems = PipelineMetadata(name="p").validate()
        assert "pipeline description is required" in problems
        assert "pipeline owner is required" in problems
        assert "pipeline must declare at least one stage" in problems

    def test_unknown_dependency_reported(self) -> None:
        metadata = PipelineMetadata(
            name="p",
            stages=[StageMetadata("a", depends_on=["ghost"])],
        )
        assert any("unknown stage 'ghost'" in problem for problem in metadata.validate())

    def test_declared_dependencies_drive_the_diagram(self, metadata: PipelineMetadata) -> None:
        diagram = metadata.to_diagram()
        assert diagram.topological_order() == ["ingest", "features", "score", "publish"]

    def test_undeclared_dependencies_chain_in_order(self) -> None:
        metadata = PipelineMetadata(
            name="p",
            stages=[
                StageMetadata("a", NodeKind.SOURCE),
                StageMetadata("b", NodeKind.SINK),
            ],
        )
        assert [(e.source, e.target) for e in metadata.to_diagram().edges] == [("a", "b")]

    def test_to_dict_is_json_serialisable(self, metadata: PipelineMetadata) -> None:
        assert json.loads(json.dumps(metadata.to_dict()))["name"] == "fraud-scoring"


class TestPipelineDocGenerator:
    """Rendering, versioning and publishing."""

    def test_generate_renders_markdown_and_diagrams(
        self, metadata: PipelineMetadata, tmp_path
    ) -> None:
        doc = PipelineDocGenerator(tmp_path).generate(metadata)
        assert doc.is_complete
        assert doc.validation_problems == ()
        assert "# Pipeline — fraud-scoring" in doc.markdown
        assert "ml-platform@example.com" in doc.markdown
        assert "```mermaid" in doc.markdown
        assert doc.mermaid.startswith("flowchart")
        assert doc.dot.startswith("digraph")

    def test_generated_markdown_lists_stages_and_io(
        self, metadata: PipelineMetadata, tmp_path
    ) -> None:
        markdown = PipelineDocGenerator(tmp_path).generate(metadata).markdown
        assert "| `ingest` | source |" in markdown
        assert "| `ledger` | dataset | s3://ledger |" in markdown
        assert "| `threshold` | `0.9` |" in markdown
        assert "- torch>=2.0" in markdown

    def test_incomplete_pipeline_surfaces_problems(self, tmp_path) -> None:
        doc = PipelineDocGenerator(tmp_path).generate(PipelineMetadata(name="p"))
        assert not doc.is_complete
        assert "⚠️" in doc.markdown

    def test_publish_writes_expected_files(self, metadata: PipelineMetadata, tmp_path) -> None:
        generator = PipelineDocGenerator(tmp_path)
        _, target = generator.generate_and_publish(metadata)
        names = {path.name for path in target.iterdir()}
        assert names == {
            "index.md",
            "pipeline.json",
            "data_flow.mmd",
            "data_flow.dot",
            "versions.json",
        }

    def test_publish_includes_model_card(self, metadata: PipelineMetadata, tmp_path) -> None:
        metadata.model_card = ModelCardBuilder("fraud-gnn").with_overview("o").build()
        _, target = PipelineDocGenerator(tmp_path).generate_and_publish(metadata)
        assert (target / "fraud_gnn_model_card.md").is_file()

    def test_versioning_is_idempotent_for_unchanged_content(
        self, metadata: PipelineMetadata, tmp_path
    ) -> None:
        generator = PipelineDocGenerator(tmp_path)
        generator.generate_and_publish(metadata)
        generator.generate_and_publish(metadata)
        assert len(generator.versions("fraud-scoring")) == 1

    def test_changed_content_appends_a_version(self, metadata: PipelineMetadata, tmp_path) -> None:
        generator = PipelineDocGenerator(tmp_path)
        generator.generate_and_publish(metadata)
        metadata.version = "1.3.0"
        generator.generate_and_publish(metadata)

        versions = generator.versions("fraud-scoring")
        assert [v.version for v in versions] == ["1.2.0", "1.3.0"]
        assert versions[0].content_hash != versions[1].content_hash

    def test_versions_of_unknown_pipeline_is_empty(self, tmp_path) -> None:
        assert PipelineDocGenerator(tmp_path).versions("ghost") == []

    def test_build_index_lists_published_pipelines(
        self, metadata: PipelineMetadata, tmp_path
    ) -> None:
        generator = PipelineDocGenerator(tmp_path)
        generator.generate_and_publish(metadata)
        index = generator.build_index()
        content = index.read_text(encoding="utf-8")
        assert "# Pipelines" in content
        assert "fraud-scoring" in content


class TestExtractMetadata:
    """Introspection of callables and objects."""

    def test_extracts_name_docstring_and_parameters(self) -> None:
        def nightly_scoring(threshold: float = 0.9, limit: int = 100) -> None:
            """Score accounts nightly.

            Longer detail that should not land in the summary.
            """

        metadata = extract_metadata(nightly_scoring)
        assert metadata.name == "nightly_scoring"
        assert metadata.description == "Score accounts nightly."
        assert metadata.parameters == {"threshold": 0.9, "limit": 100}

    def test_extracts_stages_from_attribute(self) -> None:
        class Pipeline:
            """An object pipeline."""

            stages = ["ingest", "score"]
            owner = "team@example.com"
            framework = "sklearn"

            def run(self, batch_size: int = 32) -> None:
                """Run the pipeline."""

        metadata = extract_metadata(Pipeline(), name="object-pipeline")
        assert metadata.name == "object-pipeline"
        assert [stage.name for stage in metadata.stages] == ["ingest", "score"]
        assert metadata.parameters == {"batch_size": 32}
        assert metadata.owner == "team@example.com"
        assert metadata.framework == "sklearn"

    def test_accepts_stage_metadata_instances(self) -> None:
        class Pipeline:
            stages = [StageMetadata("ingest", NodeKind.SOURCE)]

        metadata = extract_metadata(Pipeline())
        assert metadata.stages[0].kind is NodeKind.SOURCE
