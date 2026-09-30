"""Config-driven graph builder DSL (issue #739).

The point of the DSL is that an experiment can be rebuilt from the config
alone. That only holds if a spec's fingerprint tracks its content exactly,
if a typo is refused rather than ignored, and if building is a pure function
of the spec and the edge set — which is what these tests check.
"""

from __future__ import annotations

import pathlib
import random

import pytest

from astroml.features.graph.dsl import (
    SPEC_VERSION,
    EdgeSpec,
    GraphSpec,
    SpecValidationError,
    build_from_spec,
    parse_duration,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SHIPPED_SPEC = REPO_ROOT / "configs" / "graph" / "weekly_payments.yaml"

MINIMAL = {"version": 1, "name": "minimal"}

LEDGER = [
    {"src": "a", "dst": "b", "amount": 50.0, "type": "payment", "timestamp": 10},
    {"src": "a", "dst": "a", "amount": 50.0, "type": "payment", "timestamp": 11},
    {"src": "b", "dst": "c", "amount": 5.0, "type": "payment", "timestamp": 12},
    {"src": "c", "dst": "a", "amount": 500.0, "type": "trade", "timestamp": 13},
    {"src": "c", "dst": "b", "amount": 80.0, "type": "path_payment", "timestamp": 14},
]


class TestDurationParsing:
    @pytest.mark.parametrize(
        "text, seconds",
        [("30s", 30), ("15m", 900), ("24h", 86400), ("7d", 604800)],
    )
    def test_units_are_understood(self, text, seconds):
        assert parse_duration(text) == seconds

    @pytest.mark.parametrize("text", ["7q", "d7", "", "1.5d", "-3d", "seven days"])
    def test_nonsense_is_rejected(self, text):
        with pytest.raises(ValueError):
            parse_duration(text)

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            parse_duration("0d")


class TestParsing:
    def test_a_minimal_spec_takes_the_defaults(self):
        spec = GraphSpec.from_mapping(MINIMAL)

        assert spec.name == "minimal"
        assert spec.window.size == "7d"
        assert spec.edges.directed is True
        assert spec.features.max_hops == 2

    def test_every_section_is_read(self):
        spec = GraphSpec.from_mapping(
            {
                "version": 1,
                "name": "full",
                "window": {"size": "24h", "step": "1h", "t0": "2026-01-01T00:00:00Z"},
                "edges": {
                    "types": ["payment"],
                    "min_amount": 1.0,
                    "max_amount": 100.0,
                    "exclude_self_loops": False,
                    "directed": False,
                },
                "features": {"max_hops": 3, "neighbour_aggregates": False},
                "nodes": {"attributes": ["degree", "hop3_size"]},
            }
        )

        assert spec.window.step_seconds == 3600
        assert spec.window.overlapping
        assert spec.edges.types == ("payment",)
        assert spec.edges.directed is False
        assert spec.features.max_hops == 3
        assert spec.nodes.attributes == ("degree", "hop3_size")

    def test_a_step_equal_to_the_window_is_not_overlapping(self):
        spec = GraphSpec.from_mapping({**MINIMAL, "window": {"size": "1d", "step": "1d"}})

        assert not spec.window.overlapping

    def test_the_shipped_example_spec_loads(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        assert spec.name == "weekly-payments"
        assert spec.version == SPEC_VERSION

    def test_a_missing_file_is_reported_not_raised_as_an_os_error(self, tmp_path):
        with pytest.raises(SpecValidationError, match="not found"):
            GraphSpec.from_yaml(tmp_path / "absent.yaml")

    def test_malformed_yaml_is_reported(self, tmp_path):
        bad = tmp_path / "spec.yaml"
        bad.write_text("name: [unclosed\n", encoding="utf-8")

        with pytest.raises(SpecValidationError, match="not valid YAML"):
            GraphSpec.from_yaml(bad)

    def test_an_empty_file_is_reported(self, tmp_path):
        empty = tmp_path / "spec.yaml"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(SpecValidationError, match="empty"):
            GraphSpec.from_yaml(empty)

    def test_a_spec_round_trips_through_yaml(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        import yaml

        assert GraphSpec.from_mapping(yaml.safe_load(spec.to_yaml())) == spec


class TestValidation:
    def test_an_unknown_top_level_key_is_rejected(self):
        # Ignoring it would silently drop the setting the author intended.
        with pytest.raises(SpecValidationError, match="unknown key 'edge'"):
            GraphSpec.from_mapping({**MINIMAL, "edge": {}})

    def test_a_misspelled_nested_key_is_rejected(self):
        with pytest.raises(SpecValidationError, match="edges.mim_amount"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"mim_amount": 1}})

    def test_a_missing_name_is_rejected(self):
        with pytest.raises(SpecValidationError, match="name is required"):
            GraphSpec.from_mapping({"version": 1})

    def test_an_unsupported_version_is_rejected(self):
        with pytest.raises(SpecValidationError, match="unsupported spec version"):
            GraphSpec.from_mapping({**MINIMAL, "version": 99})

    def test_every_problem_is_reported_at_once(self):
        with pytest.raises(SpecValidationError) as exc:
            GraphSpec.from_mapping(
                {
                    "version": 1,
                    "name": "",
                    "typo": 1,
                    "window": {"size": "7q"},
                    "features": {"max_hops": 0},
                }
            )

        # One pass over the document, not one exception per edit.
        assert len(exc.value.errors) >= 4

    def test_an_impossible_amount_range_is_rejected(self):
        with pytest.raises(SpecValidationError, match="no edge can match"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"min_amount": 100.0, "max_amount": 1.0}})

    def test_a_step_larger_than_the_window_is_rejected(self):
        with pytest.raises(SpecValidationError, match="would skip edges"):
            GraphSpec.from_mapping({**MINIMAL, "window": {"size": "1h", "step": "1d"}})

    def test_a_negative_amount_bound_is_rejected(self):
        with pytest.raises(SpecValidationError, match="must not be negative"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"min_amount": -1}})

    def test_a_non_boolean_flag_is_rejected(self):
        with pytest.raises(SpecValidationError, match="must be a boolean"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"directed": "yes"}})

    def test_a_string_where_a_list_belongs_is_rejected(self):
        with pytest.raises(SpecValidationError, match="must be a list"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"types": "payment"}})

    def test_duplicate_edge_types_are_rejected(self):
        with pytest.raises(SpecValidationError, match="duplicates"):
            GraphSpec.from_mapping({**MINIMAL, "edges": {"types": ["payment", "payment"]}})

    def test_an_attribute_the_feature_config_cannot_produce_is_rejected(self):
        # hop3_size needs max_hops >= 3; asking for it with max_hops 2 is the
        # kind of drift a config alone would otherwise hide.
        with pytest.raises(SpecValidationError, match="hop3_size"):
            GraphSpec.from_mapping(
                {**MINIMAL, "features": {"max_hops": 2}, "nodes": {"attributes": ["hop3_size"]}}
            )

    def test_a_non_mapping_section_is_rejected(self):
        with pytest.raises(SpecValidationError, match="window must be a mapping"):
            GraphSpec.from_mapping({**MINIMAL, "window": ["7d"]})

    def test_a_non_mapping_document_is_rejected(self):
        with pytest.raises(SpecValidationError, match="mapping at the top level"):
            GraphSpec.from_mapping(["name", "x"])


class TestFingerprint:
    def test_it_is_stable_across_equivalent_documents(self):
        a = GraphSpec.from_mapping({"version": 1, "name": "x", "window": {"size": "7d"}})
        b = GraphSpec.from_mapping({"name": "x", "window": {"size": "7d"}, "version": 1})

        assert a.fingerprint() == b.fingerprint()

    def test_it_changes_when_a_setting_changes(self):
        base = GraphSpec.from_mapping(MINIMAL)
        changed = GraphSpec.from_mapping({**MINIMAL, "window": {"size": "1d"}})

        assert base.fingerprint() != changed.fingerprint()

    def test_it_changes_when_the_experiment_is_renamed(self):
        base = GraphSpec.from_mapping(MINIMAL)
        renamed = GraphSpec.from_mapping({**MINIMAL, "name": "minimal-v2"})

        # Renaming an experiment should be visible in the identifier, not
        # silently produce the same one.
        assert base.fingerprint() != renamed.fingerprint()

    def test_it_survives_a_yaml_round_trip(self, tmp_path):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)
        copy = tmp_path / "copy.yaml"
        copy.write_text(spec.to_yaml(), encoding="utf-8")

        assert GraphSpec.from_yaml(copy).fingerprint() == spec.fingerprint()


class TestEdgeFiltering:
    def test_types_restrict_which_edges_are_admitted(self):
        spec = EdgeSpec(types=("payment",))

        assert spec.accepts({"src": "a", "dst": "b", "type": "payment"})
        assert not spec.accepts({"src": "a", "dst": "b", "type": "trade"})

    def test_an_empty_type_list_admits_everything(self):
        assert EdgeSpec().accepts({"src": "a", "dst": "b", "type": "anything"})

    def test_self_loops_are_excluded_by_default(self):
        assert not EdgeSpec().accepts({"src": "a", "dst": "a"})
        assert EdgeSpec(exclude_self_loops=False).accepts({"src": "a", "dst": "a"})

    def test_amount_bounds_are_inclusive(self):
        spec = EdgeSpec(min_amount=10.0, max_amount=20.0)

        assert spec.accepts({"src": "a", "dst": "b", "amount": 10.0})
        assert spec.accepts({"src": "a", "dst": "b", "amount": 20.0})
        assert not spec.accepts({"src": "a", "dst": "b", "amount": 9.99})
        assert not spec.accepts({"src": "a", "dst": "b", "amount": 20.01})

    def test_an_edge_with_no_amount_cannot_satisfy_a_lower_bound(self):
        # Admitting it would quietly widen the spec beyond what it says.
        assert not EdgeSpec(min_amount=1.0).accepts({"src": "a", "dst": "b"})

    def test_an_edge_with_no_amount_passes_an_unfiltered_spec(self):
        assert EdgeSpec().accepts({"src": "a", "dst": "b"})


class TestBuilding:
    def test_the_spec_selects_the_edges(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        built = build_from_spec(spec, LEDGER)

        # Self-loop, sub-minimum amount and wrong type are all dropped.
        assert built.edges == (("a", "b", 50.0), ("c", "b", 80.0))
        assert built.num_rejected == 3

    def test_building_is_deterministic(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)
        shuffled = list(LEDGER)
        random.Random(5).shuffle(shuffled)

        assert build_from_spec(spec, LEDGER) == build_from_spec(spec, shuffled)

    def test_the_feature_columns_are_the_ones_the_spec_named(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        built = build_from_spec(spec, LEDGER)

        assert set(built.features.feature_names) == set(spec.nodes.attributes)

    def test_naming_no_attributes_keeps_every_feature(self):
        spec = GraphSpec.from_mapping(MINIMAL)

        built = build_from_spec(spec, LEDGER)

        assert "max_sent_amount" in built.features.feature_names

    def test_statistics_are_computed_over_the_filtered_graph(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        built = build_from_spec(spec, LEDGER, index=2)

        assert built.stats.index == 2
        assert built.stats.num_edges == 2
        assert built.stats.num_nodes == len(built.nodes)

    def test_the_result_records_which_spec_produced_it(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        built = build_from_spec(spec, LEDGER)

        # This is what makes the run reproducible from config alone.
        assert built.spec_fingerprint == spec.fingerprint()
        assert built.spec_name == spec.name

    def test_the_directed_flag_reaches_the_feature_layer(self):
        edges = [{"src": "a", "dst": "b", "amount": 1.0}, {"src": "c", "dst": "b", "amount": 1.0}]
        directed = GraphSpec.from_mapping({**MINIMAL, "edges": {"directed": True}})
        undirected = GraphSpec.from_mapping({**MINIMAL, "edges": {"directed": False}})

        assert build_from_spec(directed, edges).features.for_node("a")["hop2_size"] == 1
        assert build_from_spec(undirected, edges).features.for_node("a")["hop2_size"] == 2

    def test_the_feature_matrix_has_one_row_per_node(self):
        spec = GraphSpec.from_yaml(SHIPPED_SPEC)

        nodes, rows, columns = build_from_spec(spec, LEDGER).feature_matrix()

        assert len(rows) == len(nodes)
        assert all(len(row) == len(columns) for row in rows)

    def test_a_spec_that_admits_nothing_builds_an_empty_graph(self):
        spec = GraphSpec.from_mapping({**MINIMAL, "edges": {"types": ["nothing-matches"]}})

        built = build_from_spec(spec, LEDGER)

        assert built.nodes == ()
        assert built.num_rejected == len(LEDGER)
        assert built.stats.num_edges == 0
