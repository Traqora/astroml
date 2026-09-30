"""Multi-asset edge-type encoding (issue #733).

Two things are load-bearing here: the tensors have to satisfy what
torch_geometric's relational layers require of ``edge_type``, and the mapping
has to be stable — the same set of types must produce the same relation ids
regardless of which shard was processed first, or a model trained on one
window misreads another.
"""

from __future__ import annotations

import pytest

from astroml.features.graph.edge_types import (
    MISSING_FIELD,
    UNKNOWN_TYPE,
    EdgeTypeSpec,
    EdgeTypeVocabulary,
    UnknownEdgeTypeError,
    build_typed_edge_index,
)

LEDGER = [
    {"src": "a", "dst": "b", "asset": "USDC", "operation_type": "payment"},
    {"src": "b", "dst": "c", "asset": "XLM", "operation_type": "payment"},
    {"src": "c", "dst": "a", "asset": "USDC", "operation_type": "path_payment"},
    {"src": "a", "dst": "c", "asset": "BTC", "operation_type": "payment"},
]


class TestVocabularyConstruction:
    def test_it_covers_every_asset_present(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        assert set(vocabulary.types) == {"USDC", "XLM", "BTC"}

    def test_relation_ids_are_contiguous_from_zero(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        ids = sorted(vocabulary.id_of(t) for t in vocabulary.types)
        assert ids == list(range(vocabulary.num_relations))

    def test_a_multi_field_spec_distinguishes_operations(self):
        vocabulary = EdgeTypeVocabulary.build(
            LEDGER, EdgeTypeSpec(fields=("asset", "operation_type"))
        )

        # A USDC payment and a USDC path payment are separate relations.
        assert "USDC|payment" in vocabulary.types
        assert "USDC|path_payment" in vocabulary.types

    def test_a_missing_field_is_its_own_type(self):
        vocabulary = EdgeTypeVocabulary.build([{"src": "a", "dst": "b"}])

        # Not silently merged with a real asset name.
        assert vocabulary.types == (MISSING_FIELD,)

    def test_an_empty_field_list_is_rejected(self):
        with pytest.raises(ValueError, match="at least one field"):
            EdgeTypeVocabulary.build(LEDGER, EdgeTypeSpec(fields=()))

    def test_an_empty_separator_is_rejected(self):
        with pytest.raises(ValueError, match="separator"):
            EdgeTypeVocabulary.build(LEDGER, EdgeTypeSpec(separator=""))

    def test_an_empty_ledger_gives_an_empty_vocabulary(self):
        vocabulary = EdgeTypeVocabulary.build([])

        assert vocabulary.num_relations == 0


class TestStability:
    def test_ids_do_not_depend_on_the_order_types_were_seen(self):
        forwards = EdgeTypeVocabulary.build(LEDGER)
        backwards = EdgeTypeVocabulary.build(list(reversed(LEDGER)))

        # The bug this exists to prevent: insertion-order ids mean a model
        # trained on one shard reads another shard's relation 2 as something
        # else entirely.
        assert forwards.types == backwards.types
        assert forwards.encode(LEDGER) == backwards.encode(LEDGER)

    def test_ids_are_assigned_in_sorted_key_order(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        assert list(vocabulary.types) == sorted(vocabulary.types)

    def test_two_shards_covering_the_same_types_agree(self):
        # Both shards see USDC, XLM and BTC, but in a different order and
        # via different rows.
        first_shard = EdgeTypeVocabulary.build([LEDGER[0], LEDGER[1], LEDGER[3]])
        second_shard = EdgeTypeVocabulary.build([LEDGER[3], LEDGER[2], LEDGER[1]])

        assert first_shard.types == second_shard.types
        assert first_shard.id_of("XLM") == second_shard.id_of("XLM")

    def test_the_unknown_bucket_is_pinned_to_zero(self):
        small = EdgeTypeVocabulary.build(LEDGER[:2], allow_unknown=True)
        large = EdgeTypeVocabulary.build(LEDGER, allow_unknown=True)

        # Appending it instead would move the reserved slot every time a new
        # asset appeared, invalidating what a trained model learned about it.
        assert small.id_of(UNKNOWN_TYPE) == 0
        assert large.id_of(UNKNOWN_TYPE) == 0


class TestEncoding:
    def test_encoding_preserves_edge_order(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        encoded = vocabulary.encode(LEDGER)

        # Must line up positionally with an edge_index built from the same
        # iterable.
        assert len(encoded) == len(LEDGER)
        assert encoded[0] == vocabulary.id_of("USDC")

    def test_the_same_type_always_encodes_to_the_same_id(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        encoded = vocabulary.encode(LEDGER)
        assert encoded[0] == encoded[2]  # both USDC

    def test_ids_round_trip_back_to_type_keys(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        for edge in LEDGER:
            assert vocabulary.type_of(vocabulary.encode_one(edge)) == edge["asset"]

    def test_an_out_of_range_relation_id_is_rejected(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        with pytest.raises(IndexError, match="out of range"):
            vocabulary.type_of(vocabulary.num_relations)

    def test_counts_tally_the_edges_per_type(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        counts = vocabulary.counts(LEDGER)

        assert counts["USDC"] == 2
        assert counts["XLM"] == 1
        assert sum(counts.values()) == len(LEDGER)


class TestUnknownTypes:
    def test_an_unseen_type_is_refused_by_default(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        # An unseen relation reaching a layer sized for num_relations is a
        # crash at best; refuse it here where the message is useful.
        with pytest.raises(UnknownEdgeTypeError, match="EURC"):
            vocabulary.encode_one({"src": "x", "dst": "y", "asset": "EURC"})

    def test_the_error_names_the_known_types(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        with pytest.raises(UnknownEdgeTypeError, match="USDC"):
            vocabulary.encode_one({"asset": "EURC"})

    def test_an_unseen_type_maps_to_the_bucket_when_allowed(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER, allow_unknown=True)

        assert vocabulary.encode_one({"asset": "EURC"}) == vocabulary.id_of(UNKNOWN_TYPE)

    def test_allowing_unknowns_adds_exactly_one_relation(self):
        strict = EdgeTypeVocabulary.build(LEDGER)
        lenient = EdgeTypeVocabulary.build(LEDGER, allow_unknown=True)

        assert lenient.num_relations == strict.num_relations + 1

    def test_known_types_still_encode_normally_with_the_bucket_present(self):
        lenient = EdgeTypeVocabulary.build(LEDGER, allow_unknown=True)

        assert lenient.type_of(lenient.encode_one(LEDGER[0])) == "USDC"


class TestTorchGeometricCompatibility:
    def test_edge_type_is_a_one_dimensional_int64_tensor(self):
        torch = pytest.importorskip("torch")
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        edge_type = vocabulary.as_tensor(LEDGER)

        # RGCNConv indexes its relation weights with this; anything but a 1-D
        # int64 tensor fails inside the layer.
        assert edge_type.dtype == torch.int64
        assert edge_type.dim() == 1
        assert edge_type.size(0) == len(LEDGER)

    def test_values_lie_within_num_relations(self):
        pytest.importorskip("torch")
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        edge_type = vocabulary.as_tensor(LEDGER)

        assert int(edge_type.min()) >= 0
        assert int(edge_type.max()) < vocabulary.num_relations

    def test_the_typed_index_pairs_shapes_correctly(self):
        torch = pytest.importorskip("torch")

        typed = build_typed_edge_index(LEDGER)
        edge_index, edge_type = typed.to_tensors()

        assert edge_index.shape == (2, len(LEDGER))
        assert edge_type.shape == (len(LEDGER),)
        assert edge_index.dtype == torch.int64

    def test_it_drives_a_relational_layer(self):
        pytest.importorskip("torch")
        pyg_nn = pytest.importorskip("torch_geometric.nn")
        import torch

        typed = build_typed_edge_index(LEDGER)
        edge_index, edge_type = typed.to_tensors()

        conv = pyg_nn.RGCNConv(4, 8, num_relations=typed.num_relations)
        out = conv(torch.randn(len(typed.nodes), 4), edge_index, edge_type)

        assert out.shape == (len(typed.nodes), 8)

    def test_a_vocabulary_without_the_bucket_sizes_the_layer_exactly(self):
        typed = build_typed_edge_index(LEDGER)

        assert typed.num_relations == len(set(edge["asset"] for edge in LEDGER))


class TestTypedEdgeIndex:
    def test_nodes_are_sorted_and_indices_address_them(self):
        typed = build_typed_edge_index(LEDGER)

        assert list(typed.nodes) == sorted(typed.nodes)
        for row, col in zip(*typed.edge_index):
            assert 0 <= row < len(typed.nodes)
            assert 0 <= col < len(typed.nodes)

    def test_edge_type_lines_up_with_edge_index(self):
        typed = build_typed_edge_index(LEDGER)

        assert len(typed.edge_type) == len(typed.edge_index[0])

    def test_the_output_is_order_independent(self):
        forwards = build_typed_edge_index(LEDGER)
        backwards = build_typed_edge_index(list(reversed(LEDGER)))

        assert forwards.edge_index == backwards.edge_index
        assert forwards.edge_type == backwards.edge_type

    def test_an_existing_vocabulary_is_reused_rather_than_rebuilt(self):
        training = EdgeTypeVocabulary.build(LEDGER)
        evaluation_edges = [LEDGER[1]]

        typed = build_typed_edge_index(evaluation_edges, vocabulary=training)

        # Rebuilding from the eval slice would give XLM a different id than
        # training used.
        assert typed.num_relations == training.num_relations
        assert typed.edge_type == (training.id_of("XLM"),)

    def test_an_edge_without_endpoints_is_rejected(self):
        with pytest.raises(ValueError, match="missing src or dst"):
            build_typed_edge_index([{"asset": "USDC"}])


class TestSnapshotEdgeCompatibility:
    def test_snapshot_edge_dataclasses_are_accepted(self):
        from astroml.features.graph.snapshot import Edge

        vocabulary = EdgeTypeVocabulary.build(
            [Edge(src="a", dst="b", timestamp=1, amount=1.0, asset="USDC")]
        )

        assert vocabulary.types == ("USDC",)

    def test_a_positional_edge_without_an_asset_still_constructs(self):
        from astroml.features.graph.snapshot import Edge

        vocabulary = EdgeTypeVocabulary.build([Edge("a", "b", 1)])

        assert vocabulary.types == (MISSING_FIELD,)


class TestPersistence:
    def test_a_vocabulary_round_trips_through_json(self):
        vocabulary = EdgeTypeVocabulary.build(
            LEDGER, EdgeTypeSpec(fields=("asset", "operation_type")), allow_unknown=True
        )

        restored = EdgeTypeVocabulary.from_json(vocabulary.to_json())

        assert restored == vocabulary
        assert restored.encode(LEDGER) == vocabulary.encode(LEDGER)

    def test_the_spec_survives_the_round_trip(self):
        vocabulary = EdgeTypeVocabulary.build(
            LEDGER, EdgeTypeSpec(fields=("asset", "operation_type"), separator="::")
        )

        restored = EdgeTypeVocabulary.from_dict(vocabulary.to_dict())

        assert restored.spec.fields == ("asset", "operation_type")
        assert restored.spec.separator == "::"

    def test_a_reloaded_vocabulary_keeps_its_ids(self):
        vocabulary = EdgeTypeVocabulary.build(LEDGER)

        restored = EdgeTypeVocabulary.from_json(vocabulary.to_json())

        # This is what makes a saved model usable: the ids it trained on have
        # to survive the trip to disk.
        for type_key in vocabulary.types:
            assert restored.id_of(type_key) == vocabulary.id_of(type_key)
