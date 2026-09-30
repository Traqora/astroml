"""Tests for astroml.governance.model_card.

Covers ModelCard serialization (to_dict/to_json/to_markdown) and the
ModelCardGenerator convenience API, including generation from a
RiskAssessment.
"""

from __future__ import annotations

import json

import pytest

from astroml.governance.compliance import RiskAssessment, RiskCategory, RiskFinding, RiskLevel
from astroml.governance.model_card import (
    EthicalConsiderations,
    FairnessEvaluation,
    IntendedUse,
    ModelCard,
    ModelCardGenerator,
    ModelDetails,
    ModelType,
    PerformanceMetrics,
    TrainingData,
)


@pytest.fixture
def sample_card() -> ModelCard:
    return ModelCard(
        model_details=ModelDetails(
            name="Fraud Detector",
            version="2.0.0",
            model_type=ModelType.ANOMALY_DETECTION,
            description="Graph-based fraud detection model",
            authors=["ML Team"],
            framework="pytorch",
            framework_version="2.1.0",
            license="Apache-2.0",
        ),
        intended_use=IntendedUse(
            primary_use="Detect fraudulent transactions on Stellar network",
            primary_users=["Risk analysts"],
            out_of_scope_uses=["Credit scoring"],
            limitations=["Not validated on non-Stellar chains"],
        ),
        training_data=TrainingData(data_sources=["horizon_stream"], data_size=1_000_000),
        performance=PerformanceMetrics(accuracy=0.95, f1_score=0.93),
        fairness=FairnessEvaluation(methodology="group parity", findings=["No disparity found"]),
        ethical=EthicalConsiderations(
            biases_identified=["Under-represented low-volume accounts"],
            mitigations=["Reweighted training sample"],
        ),
        caveats=["Retrain quarterly"],
        references=["https://example.com/paper"],
    )


class TestModelCardToDict:
    def test_to_dict_contains_all_top_level_sections(self, sample_card: ModelCard):
        data = sample_card.to_dict()
        for key in (
            "card_id",
            "version",
            "generated_at",
            "model_details",
            "intended_use",
            "training_data",
            "performance",
            "fairness",
            "ethical_considerations",
            "caveats",
            "references",
            "quantitative_analysis",
        ):
            assert key in data

    def test_model_details_serialized_correctly(self, sample_card: ModelCard):
        details = sample_card.to_dict()["model_details"]
        assert details["name"] == "Fraud Detector"
        assert details["model_type"] == "anomaly_detection"
        assert details["authors"] == ["ML Team"]

    def test_performance_metrics_only_include_set_values(self, sample_card: ModelCard):
        performance = sample_card.to_dict()["performance"]
        assert performance == {"accuracy": 0.95, "f1_score": 0.93}

    def test_additional_metrics_are_merged_in(self):
        perf = PerformanceMetrics(accuracy=0.9, additional_metrics={"custom_score": 0.5})
        assert perf.to_dict() == {"accuracy": 0.9, "custom_score": 0.5}

    def test_to_dict_is_json_serializable(self, sample_card: ModelCard):
        json.dumps(sample_card.to_dict(), default=str)

    def test_created_date_none_serializes_to_none(self):
        card = ModelCard(model_details=ModelDetails(name="m", version="1"))
        data = card.to_dict()
        assert data["model_details"]["created_date"] is None
        assert data["model_details"]["last_updated"] is None


class TestModelCardToJson:
    def test_to_json_writes_valid_file(self, sample_card: ModelCard, tmp_path):
        out_path = tmp_path / "card.json"
        sample_card.to_json(out_path)

        written = json.loads(out_path.read_text())
        assert written["model_details"]["name"] == "Fraud Detector"


class TestModelCardToMarkdown:
    def test_to_markdown_includes_key_sections(self, sample_card: ModelCard):
        markdown = sample_card.to_markdown()
        assert "# Model Card: Fraud Detector v2.0.0" in markdown
        assert "## Model Details" in markdown
        assert "## Intended Use" in markdown
        assert "## Performance Metrics" in markdown
        assert "## Fairness Evaluation" in markdown
        assert "## Ethical Considerations" in markdown
        assert "## Caveats and Recommendations" in markdown
        assert "## References" in markdown

    def test_to_markdown_omits_empty_optional_sections(self):
        card = ModelCard(model_details=ModelDetails(name="m", version="1"))
        markdown = card.to_markdown()
        assert "## Caveats and Recommendations" not in markdown
        assert "## References" not in markdown

    def test_to_markdown_writes_file_when_path_given(self, sample_card: ModelCard, tmp_path):
        out_path = tmp_path / "card.md"
        content = sample_card.to_markdown(out_path)
        assert out_path.read_text() == content

    def test_to_markdown_handles_no_performance_metrics(self):
        card = ModelCard(model_details=ModelDetails(name="m", version="1"))
        markdown = card.to_markdown()
        assert "## Performance Metrics" in markdown


class TestModelCardGenerator:
    def test_generate_populates_all_sections(self):
        gen = ModelCardGenerator()
        card = gen.generate(
            model_name="Fraud Detector",
            model_version="2.0.0",
            model_type=ModelType.ANOMALY_DETECTION,
            description="desc",
            authors=["ML Team"],
            performance={"accuracy": 0.95, "custom_metric": 0.8},
            training_data_size=500,
            training_data_sources=["stream"],
            primary_use="fraud detection",
            biases=["bias-a"],
            mitigations=["mitigation-a"],
            caveats=["caveat-a"],
            references=["ref-a"],
        )

        assert card.model_details.name == "Fraud Detector"
        assert card.model_details.created_date is not None
        assert card.performance.accuracy == 0.95
        assert card.performance.additional_metrics == {"custom_metric": 0.8}
        assert card.training_data.data_size == 500
        assert card.ethical.biases_identified == ["bias-a"]
        assert card.ethical.mitigations == ["mitigation-a"]
        assert card.caveats == ["caveat-a"]
        assert card.references == ["ref-a"]

    def test_generate_with_no_optional_args_uses_safe_defaults(self):
        gen = ModelCardGenerator()
        card = gen.generate(model_name="m", model_version="1")
        assert card.model_details.authors == []
        assert card.performance.to_dict() == {}
        assert card.training_data.data_sources == []

    def test_generate_from_risk_assessment_splits_fairness_vs_caveats(self):
        gen = ModelCardGenerator()
        assessment = RiskAssessment(
            model_id="m",
            findings=[
                RiskFinding(
                    category=RiskCategory.FAIRNESS,
                    level=RiskLevel.MEDIUM,
                    description="Disparity in group A",
                    mitigation="Reweight training data",
                ),
                RiskFinding(
                    category=RiskCategory.SECURITY,
                    level=RiskLevel.HIGH,
                    description="Model susceptible to adversarial input",
                ),
            ],
        )

        card = gen.generate_from_risk_assessment(
            model_name="Fraud Detector",
            model_version="2.0.0",
            risk_assessment=assessment,
            model_type=ModelType.ANOMALY_DETECTION,
        )

        assert "Disparity in group A" in card.ethical.biases_identified
        assert "Reweight training data" in card.ethical.mitigations
        assert any("adversarial input" in c for c in card.caveats)
        assert any(c.startswith("[HIGH]") for c in card.caveats)
