"""Tests for multi-modal model support (#631)."""

from __future__ import annotations

import numpy as np
import pytest

from astroml.preprocessing.multimodal import MultiModalSample, collate_batch
from astroml.training.multimodal import (
    AttentionFusion,
    ConcatenationFusion,
    CrossModalRetriever,
    EncoderRegistry,
    FusionMethod,
    GatedFusion,
    ImageEncoder,
    ImageEncoderType,
    MaxFusion,
    MeanFusion,
    Modality,
    MultiModalConfig,
    MultiModalDataBatch,
    MultiModalEncoder,
    MultiModalPipeline,
    SumFusion,
    TabularEncoder,
    TextEncoder,
    TextEncoderType,
    create_fusion,
)

# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------


class TestEncoders:
    def test_text_encoder_shape(self) -> None:
        encoder = TextEncoder(encoder_type=TextEncoderType.BERT)
        texts = ["hello world", "this is a test"]
        embeddings = encoder.encode(texts)
        assert embeddings.shape == (2, 768)
        assert np.all(np.isfinite(embeddings))

    def test_text_encoder_normalized(self) -> None:
        encoder = TextEncoder()
        embeddings = encoder.encode(["a", "b", "c"])
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(3), decimal=4)

    def test_image_encoder_shape(self) -> None:
        encoder = ImageEncoder(encoder_type=ImageEncoderType.RESNET18)
        images = np.random.randn(4, 3, 224, 224).astype(np.float32)
        embeddings = encoder.encode(images)
        assert embeddings.shape == (4, 512)

    def test_image_encoder_3d_input(self) -> None:
        encoder = ImageEncoder()
        image = np.random.randn(3, 224, 224).astype(np.float32)
        embeddings = encoder.encode(image)
        assert embeddings.shape == (1, 2048)

    def test_tabular_encoder_shape(self) -> None:
        encoder = TabularEncoder(input_dim=10, embedding_dim=128)
        data = np.random.randn(32, 10).astype(np.float32)
        embeddings = encoder.encode(data)
        assert embeddings.shape == (32, 128)

    def test_tabular_encoder_normalized(self) -> None:
        encoder = TabularEncoder(input_dim=5, embedding_dim=64)
        data = np.random.randn(8, 5).astype(np.float32)
        embeddings = encoder.encode(data)
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(8), decimal=4)

    def test_all_encoder_modalities(self) -> None:
        assert TextEncoder().modality == Modality.TEXT
        assert ImageEncoder().modality == Modality.IMAGE
        assert TabularEncoder(input_dim=4).modality == Modality.TABULAR

    def test_encoder_registry(self) -> None:
        registry = EncoderRegistry()
        text_enc = TextEncoder()
        img_enc = ImageEncoder()
        registry.register(text_enc)
        registry.register(img_enc)

        assert registry.get_encoder(Modality.TEXT) is text_enc
        assert registry.get_encoder(Modality.AUDIO) is None

        text_emb = registry.encode(Modality.TEXT, ["hello"])
        assert text_emb.shape == (1, 768)

        dims = registry.get_embedding_dims()
        assert Modality.TEXT in dims
        assert Modality.IMAGE in dims

    def test_registry_encode_batch(self) -> None:
        registry = EncoderRegistry()
        registry.register(TextEncoder())
        registry.register(TabularEncoder(input_dim=3, embedding_dim=64))

        batch = {
            Modality.TEXT: ["a", "b"],
            Modality.TABULAR: np.random.randn(2, 3).astype(np.float32),
        }
        results = registry.encode_batch(batch)
        assert Modality.TEXT in results
        assert Modality.TABULAR in results
        assert results[Modality.TEXT].shape == (2, 768)
        assert results[Modality.TABULAR].shape == (2, 64)


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


class TestFusion:
    @pytest.fixture
    def embeddings(self) -> dict[Modality, np.ndarray]:
        batch = 4
        return {
            Modality.TEXT: np.random.randn(batch, 768).astype(np.float32),
            Modality.IMAGE: np.random.randn(batch, 512).astype(np.float32),
        }

    def test_concat_fusion(self, embeddings: dict[Modality, np.ndarray]) -> None:
        fusion = ConcatenationFusion(output_dim=256, hidden_dim=512)
        fused = fusion.fuse(embeddings)
        assert fused.shape == (4, 256)
        assert fusion.get_output_dim() == 256

    def test_attention_fusion(self, embeddings: dict[Modality, np.ndarray]) -> None:
        fusion = AttentionFusion(output_dim=256, num_heads=4)
        fused = fusion.fuse(embeddings)
        assert fused.shape == (4, 256)
        assert np.all(np.isfinite(fused))

    def test_gated_fusion(self, embeddings: dict[Modality, np.ndarray]) -> None:
        fusion = GatedFusion(output_dim=256)
        fusion.set_gate(Modality.TEXT, 0.7)
        fusion.set_gate(Modality.IMAGE, 0.3)
        fused = fusion.fuse(embeddings)
        assert fused.shape == (4, 256)
        gates = fusion.get_gates()
        assert Modality.TEXT in gates
        assert Modality.IMAGE in gates

    def test_sum_fusion(self, embeddings: dict[Modality, np.ndarray]) -> None:
        # Sum requires same dim → project both to 256
        proj_emb = {
            Modality.TEXT: np.random.randn(4, 256).astype(np.float32),
            Modality.IMAGE: np.random.randn(4, 256).astype(np.float32),
        }
        fusion = SumFusion()
        fused = fusion.fuse(proj_emb)
        assert fused.shape == (4, 256)

    def test_mean_fusion(self) -> None:
        emb = {
            Modality.TEXT: np.random.randn(4, 128).astype(np.float32),
            Modality.IMAGE: np.random.randn(4, 128).astype(np.float32),
        }
        fusion = MeanFusion()
        fused = fusion.fuse(emb)
        assert fused.shape == (4, 128)

    def test_max_fusion(self) -> None:
        emb = {
            Modality.TEXT: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            Modality.IMAGE: np.array([[5.0, 1.0], [0.0, 6.0]], dtype=np.float32),
        }
        fusion = MaxFusion()
        fused = fusion.fuse(emb)
        assert fused[0, 0] == 5.0
        assert fused[1, 1] == 6.0

    def test_create_fusion_factory(self) -> None:
        fusion = create_fusion(FusionMethod.CONCAT, output_dim=128)
        assert isinstance(fusion, ConcatenationFusion)
        assert fusion.get_output_dim() == 128

        fusion = create_fusion(FusionMethod.SUM)
        assert isinstance(fusion, SumFusion)

    def test_single_modality_fusion(self) -> None:
        fusion = ConcatenationFusion(output_dim=128)
        emb = {Modality.TEXT: np.random.randn(4, 768).astype(np.float32)}
        fused = fusion.fuse(emb)
        assert fused.shape == (4, 128)

    def test_cross_modal_retriever(self) -> None:
        retriever = CrossModalRetriever()
        # Index 10 text embeddings
        text_emb = np.random.randn(10, 256).astype(np.float32)
        retriever.index(Modality.TEXT, text_emb)
        # Query with a random embedding
        query = np.random.randn(256).astype(np.float32)
        indices, sims = retriever.retrieve(query, Modality.TEXT, top_k=3)
        assert len(indices) == 3
        assert len(sims) == 3
        assert all(0 <= i < 10 for i in indices)
        assert all(-1 <= s <= 1 for s in sims)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class TestPipeline:
    def test_pipeline_setup(self) -> None:
        config = MultiModalConfig(
            enabled_modalities=[Modality.TEXT, Modality.TABULAR],
            fusion_method=FusionMethod.CONCAT,
            cross_modal_retrieval=False,
        )
        pipeline = MultiModalPipeline(config)
        dims = pipeline.get_embedding_dims()
        assert Modality.TEXT in dims
        assert Modality.TABULAR in dims
        assert Modality.IMAGE not in dims
        assert pipeline.get_fusion_output_dim() == config.fusion_output_dim

    def test_pipeline_train(self) -> None:
        pipeline = MultiModalPipeline(
            MultiModalConfig(
                enabled_modalities=[Modality.TEXT, Modality.TABULAR],
                epochs=3,
                tabular_input_dim=10,
            )
        )
        batches = [
            MultiModalDataBatch(
                text_inputs=["sample 1", "sample 2"],
                tabular_inputs=np.random.randn(2, 10).astype(np.float32),
                labels=np.array([0, 1]),
            )
            for _ in range(5)
        ]
        result = pipeline.train(batches)
        assert result["epochs_completed"] == 3
        assert result["final_loss"] is not None
        assert len(result["loss_history"]) == 3

    def test_pipeline_infer(self) -> None:
        pipeline = MultiModalPipeline(
            MultiModalConfig(enabled_modalities=[Modality.TEXT])
        )
        batch = MultiModalDataBatch(text_inputs=["hello", "world"])
        fused = pipeline.infer(batch)
        assert fused.shape == (2, pipeline.get_fusion_output_dim())

    def test_pipeline_cross_modal(self) -> None:
        pipeline = MultiModalPipeline(
            MultiModalConfig(
                enabled_modalities=[Modality.TEXT, Modality.TABULAR],
                cross_modal_retrieval=True,
                tabular_input_dim=10,
            )
        )
        # Index some tabular embeddings
        tab_emb = np.random.randn(5, pipeline.get_fusion_output_dim()).astype(np.float32)
        pipeline.index_modality(Modality.TABULAR, tab_emb)

        query = np.random.randn(pipeline.get_fusion_output_dim()).astype(np.float32)
        indices, sims = pipeline.retrieve_cross_modal(query, Modality.TABULAR, top_k=3)
        assert len(indices) == 3


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_multimodal_sample(self) -> None:
        sample = MultiModalSample(
            text="hello",
            image=np.random.randn(3, 224, 224).astype(np.float32),
            tabular=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            label=1,
            sample_id="s1",
        )
        assert sample.text == "hello"
        assert sample.label == 1

    def test_collate_batch(self) -> None:
        samples = [
            MultiModalSample(text="a", tabular=np.array([1.0], dtype=np.float32), label=0),
            MultiModalSample(text="b", tabular=np.array([2.0], dtype=np.float32), label=1),
        ]
        batch = collate_batch(samples)
        assert batch["text_inputs"] == ["a", "b"]
        assert batch["tabular_inputs"] is not None
        assert batch["tabular_inputs"].shape == (2, 1)
        assert batch["labels"].tolist() == [0, 1]
        assert batch["image_inputs"] is None