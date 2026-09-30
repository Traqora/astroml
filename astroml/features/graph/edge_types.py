"""Multi-asset edge-type encoding for snapshots — issue #733.

Relational GNN layers (``RGCNConv``, ``FastRGCNConv``, ``HeteroConv``) take an
``edge_type`` tensor alongside ``edge_index``: one ``int64`` per edge, values
in ``[0, num_relations)``. This module turns the asset and operation type
carried on a Stellar transaction into exactly that.

The mapping is the whole problem. A vocabulary built by "assign the next id
each time I see a new type" gives different ids for the same data depending
on which window is processed first — so a model trained on Monday's shard
reads Tuesday's ``edge_type=3`` as a different relation entirely, silently.
:class:`EdgeTypeVocabulary` assigns ids in **sorted order of the type key**,
never insertion order, so the same set of types always produces the same ids
on any machine, in any shard order.

Two policies for a type that was not in the training data:
``allow_unknown=False`` (the default) refuses it, because an unseen relation
reaching a layer sized for ``num_relations`` is a crash at best and a wrong
answer at worst; ``allow_unknown=True`` maps it to a reserved bucket at id 0.

Encoding is a dict lookup per edge, so the cost is linear in edges with no
per-type scan.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    import torch

#: Reserved key for edges whose type was not seen when the vocabulary was
#: built. Always id 0 when ``allow_unknown`` is set, so the reserved slot does
#: not move as the vocabulary grows.
UNKNOWN_TYPE = "<unknown>"

#: Placeholder for an edge field that is absent, so a missing asset is a type
#: in its own right rather than colliding with a real one.
MISSING_FIELD = "<none>"

_DEFAULT_FIELDS = ("asset",)
_SEPARATOR = "|"


class UnknownEdgeTypeError(KeyError):
    """Raised when encoding an edge whose type is not in the vocabulary."""


@dataclass(frozen=True)
class EdgeTypeSpec:
    """Which edge fields make up the type key.

    ``fields=("asset",)`` gives one relation per asset;
    ``fields=("asset", "operation_type")`` distinguishes a USDC payment from
    a USDC path payment. The fields are read in the order given and joined,
    so the spec fully determines the key.
    """

    fields: tuple[str, ...] = _DEFAULT_FIELDS
    separator: str = _SEPARATOR

    def validate(self) -> None:
        if not self.fields:
            raise ValueError("EdgeTypeSpec.fields must name at least one field")
        if not self.separator:
            raise ValueError("EdgeTypeSpec.separator must be a non-empty string")

    def key_for(self, edge: Any) -> str:
        """The type key of one edge."""
        parts: list[str] = []
        for field_name in self.fields:
            if isinstance(edge, Mapping):
                value = edge.get(field_name)
            else:
                value = getattr(edge, field_name, None)
            parts.append(MISSING_FIELD if value is None else str(value))
        return self.separator.join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {"fields": list(self.fields), "separator": self.separator}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EdgeTypeSpec:
        return cls(
            fields=tuple(payload.get("fields", _DEFAULT_FIELDS)),
            separator=payload.get("separator", _SEPARATOR),
        )


class EdgeTypeVocabulary:
    """A stable type-key to relation-id mapping.

    Ids are contiguous from 0, which is what ``num_relations`` in a
    ``RGCNConv`` means, and are assigned in sorted key order so they do not
    depend on how the data was sharded or which snapshot was seen first.
    """

    def __init__(
        self,
        types: Sequence[str],
        spec: EdgeTypeSpec | None = None,
        allow_unknown: bool = False,
    ) -> None:
        self.spec = spec or EdgeTypeSpec()
        self.spec.validate()
        self.allow_unknown = allow_unknown

        known = sorted({str(t) for t in types} - {UNKNOWN_TYPE})
        # The unknown bucket is pinned to 0 rather than appended, so adding a
        # new asset later shifts nothing that a trained model has learned
        # about the reserved slot.
        ordered = ([UNKNOWN_TYPE] if allow_unknown else []) + known

        self._types: tuple[str, ...] = tuple(ordered)
        self._ids: dict[str, int] = {name: index for index, name in enumerate(ordered)}

    # -- construction ----------------------------------------------------

    @classmethod
    def build(
        cls,
        edges: Iterable[Any],
        spec: EdgeTypeSpec | None = None,
        allow_unknown: bool = False,
    ) -> EdgeTypeVocabulary:
        """Build a vocabulary covering every type present in ``edges``."""
        spec = spec or EdgeTypeSpec()
        spec.validate()
        return cls(
            types=sorted({spec.key_for(edge) for edge in edges}),
            spec=spec,
            allow_unknown=allow_unknown,
        )

    # -- inspection ------------------------------------------------------

    @property
    def num_relations(self) -> int:
        """The ``num_relations`` a relational layer must be sized for."""
        return len(self._types)

    @property
    def types(self) -> tuple[str, ...]:
        """Type keys, indexed by their relation id."""
        return self._types

    def __len__(self) -> int:
        return len(self._types)

    def __contains__(self, type_key: object) -> bool:
        return str(type_key) in self._ids

    def id_of(self, type_key: str) -> int:
        """Relation id for a type key."""
        try:
            return self._ids[str(type_key)]
        except KeyError:
            if self.allow_unknown:
                return self._ids[UNKNOWN_TYPE]
            raise UnknownEdgeTypeError(
                f"edge type {type_key!r} is not in this vocabulary "
                f"(known: {', '.join(self._types) or 'none'}); "
                f"rebuild it or construct with allow_unknown=True"
            ) from None

    def type_of(self, relation_id: int) -> str:
        """The type key behind a relation id."""
        if not 0 <= relation_id < len(self._types):
            raise IndexError(
                f"relation id {relation_id} out of range for {len(self._types)} relations"
            )
        return self._types[relation_id]

    # -- encoding --------------------------------------------------------

    def encode_one(self, edge: Any) -> int:
        return self.id_of(self.spec.key_for(edge))

    def encode(self, edges: Iterable[Any]) -> list[int]:
        """Relation id per edge, in the order given.

        The order is preserved deliberately: the result lines up positionally
        with the ``edge_index`` built from the same iterable.
        """
        return [self.encode_one(edge) for edge in edges]

    def as_tensor(self, edges: Iterable[Any]) -> "torch.Tensor":
        """The ``edge_type`` tensor a relational layer expects.

        A 1-D ``int64`` tensor of length E with values in
        ``[0, num_relations)``. Torch is imported lazily so the mapping stays
        usable — and testable — in a process that has no torch.
        """
        import torch

        return torch.tensor(self.encode(edges), dtype=torch.long)

    def counts(self, edges: Iterable[Any]) -> dict[str, int]:
        """How many edges fall into each type, for reporting and sanity checks."""
        tally = {name: 0 for name in self._types}
        for edge in edges:
            tally[self.type_of(self.encode_one(edge))] += 1
        return tally

    # -- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "allow_unknown": self.allow_unknown,
            "types": list(self._types),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EdgeTypeVocabulary:
        return cls(
            types=list(payload["types"]),
            spec=EdgeTypeSpec.from_dict(payload.get("spec", {})),
            allow_unknown=bool(payload.get("allow_unknown", False)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> EdgeTypeVocabulary:
        return cls.from_dict(json.loads(text))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EdgeTypeVocabulary):
            return NotImplemented
        return (
            self._types == other._types
            and self.spec == other.spec
            and self.allow_unknown == other.allow_unknown
        )

    def __repr__(self) -> str:
        return (
            f"EdgeTypeVocabulary(num_relations={self.num_relations}, "
            f"fields={self.spec.fields}, allow_unknown={self.allow_unknown})"
        )


@dataclass(frozen=True)
class TypedEdgeIndex:
    """An ``edge_index``/``edge_type`` pair plus the vocabulary behind it."""

    nodes: tuple[str, ...]
    edge_index: tuple[tuple[int, ...], tuple[int, ...]]
    edge_type: tuple[int, ...]
    vocabulary: EdgeTypeVocabulary

    @property
    def num_relations(self) -> int:
        return self.vocabulary.num_relations

    def to_tensors(self) -> tuple["torch.Tensor", "torch.Tensor"]:
        """``(edge_index, edge_type)`` as the tensors PyG expects."""
        import torch

        edge_index = torch.tensor(
            [list(self.edge_index[0]), list(self.edge_index[1])], dtype=torch.long
        )
        return edge_index, torch.tensor(list(self.edge_type), dtype=torch.long)


def build_typed_edge_index(
    edges: Iterable[Any],
    vocabulary: EdgeTypeVocabulary | None = None,
    spec: EdgeTypeSpec | None = None,
    allow_unknown: bool = False,
) -> TypedEdgeIndex:
    """Build ``edge_index`` and ``edge_type`` together from raw edges.

    Args:
        edges: Edge records — ``Edge`` instances or mappings.
        vocabulary: An existing vocabulary to encode against. Pass the one
            from training when encoding evaluation data, so the relation ids
            mean the same thing; omit it to build a fresh one.
        spec / allow_unknown: Used only when building a fresh vocabulary.

    Returns:
        A :class:`TypedEdgeIndex`. Nodes are sorted and edges canonically
        ordered, so the same edge set always yields the same tensors.
    """
    materialised = list(edges)
    if vocabulary is None:
        vocabulary = EdgeTypeVocabulary.build(materialised, spec=spec, allow_unknown=allow_unknown)

    rows: list[tuple[str, str, int]] = []
    nodes: set[str] = set()
    for edge in materialised:
        if isinstance(edge, Mapping):
            src = edge.get("src", edge.get("source"))
            dst = edge.get("dst", edge.get("destination"))
        else:
            src = getattr(edge, "src", None)
            dst = getattr(edge, "dst", None)
        if src is None or dst is None:
            raise ValueError(f"edge is missing src or dst: {edge!r}")
        src, dst = str(src), str(dst)
        nodes.add(src)
        nodes.add(dst)
        rows.append((src, dst, vocabulary.encode_one(edge)))

    rows.sort()
    ordered_nodes = tuple(sorted(nodes))
    index_of = {node: i for i, node in enumerate(ordered_nodes)}

    return TypedEdgeIndex(
        nodes=ordered_nodes,
        edge_index=(
            tuple(index_of[src] for src, _, _ in rows),
            tuple(index_of[dst] for _, dst, _ in rows),
        ),
        edge_type=tuple(relation for _, _, relation in rows),
        vocabulary=vocabulary,
    )


__all__ = [
    "MISSING_FIELD",
    "UNKNOWN_TYPE",
    "EdgeTypeSpec",
    "EdgeTypeVocabulary",
    "TypedEdgeIndex",
    "UnknownEdgeTypeError",
    "build_typed_edge_index",
]
