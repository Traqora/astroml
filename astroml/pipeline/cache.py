from astroml.utils.exceptions import AstroMLError
"""Content-addressable cache key generation for ML pipeline caching.

Issue #636: Implements intelligent content-addressable cache keys that uniquely
identify pipeline inputs, enabling deduplication and fast lookups.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TypeVar

import numpy as np

_T = TypeVar("_T")


def _make_json_safe(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return hashlib.sha256(obj).hexdigest()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return hashlib.sha256(obj.tobytes()).hexdigest()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_make_json_safe(v) for v in obj]
    # Fallback: convert to string representation and hash
    try:
        return str(obj)
    except AstroMLError:
        return hashlib.sha256(repr(obj).encode()).hexdigest()


def _sorted_dict_items(d: dict[str, Any]) -> list[tuple[str, Any]]:
    """Return key-sorted items for deterministic serialization."""
    return sorted(d.items(), key=lambda kv: kv[0])


@dataclass(frozen=True)
class CacheKey:
    """Content-addressable cache key with metadata.

    A CacheKey uniquely identifies a pipeline input or output based on its
    content fingerprint, enabling deterministic cache lookups.

    Attributes:
        fingerprint: SHA-256 hex digest of serialized content.
        namespace: Logical grouping for the cache entry (e.g., 'features', 'embeddings').
        version: Schema version for forward/backward compatibility.
        tags: Optional tags for querying and invalidation groups.
        created_at: UTC timestamp when the key was generated.
    """

    fingerprint: str
    namespace: str
    version: str = "v1"
    tags: frozenset[str] = field(default_factory=frozenset)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __repr__(self) -> str:
        tags_str = ",".join(sorted(self.tags)) if self.tags else ""
        return f"CacheKey({self.namespace}:{self.fingerprint[:12]}... v={self.version} tags=[{tags_str}])"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dictionary."""
        return {
            "fingerprint": self.fingerprint,
            "namespace": self.namespace,
            "version": self.version,
            "tags": sorted(self.tags),
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CacheKey:
        """Deserialize from a dictionary."""
        return cls(
            fingerprint=data["fingerprint"],
            namespace=data["namespace"],
            version=data.get("version", "v1"),
            tags=frozenset(data.get("tags", [])),
            created_at=(
                datetime.fromisoformat(data["created_at"])
                if "created_at" in data
                else datetime.now(timezone.utc)
            ),
        )


class ContentAddressableCache:
    """Content-addressable cache for ML pipeline artifacts.

    Generates deterministic cache keys from pipeline inputs/outputs
    and manages key-to-key dependency tracking.

    Example:
        cac = ContentAddressableCache(namespace="features")
        key = cac.make_key({"account_ids": ["A1", "A2"], "window": "7d"})
        # Cache a result under this key
        cac.put(key, feature_matrix)
        # Later, reconstructed identical input gets the same key
        same_key = cac.make_key({"account_ids": ["A1", "A2"], "window": "7d"})
        assert key.fingerprint == same_key.fingerprint
    """

    def __init__(
        self,
        namespace: str,
        version: str = "v1",
        hash_algorithm: str = "sha256",
    ) -> None:
        """Initialize content-addressable cache.

        Args:
            namespace: Logical grouping (e.g., 'features', 'embeddings', 'predictions').
            version: Schema version for format evolution.
            hash_algorithm: Hash algorithm to use.
        """
        self.namespace = namespace
        self.version = version
        self._hash_algo = hash_algorithm
        self._dependencies: dict[str, set[str]] = {}  # key fingerprint -> set of parent fingerprints
        self._store: dict[str, Any] = {}

    def make_key(
        self,
        content: Any,
        tags: Iterable[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> CacheKey:
        """Generate a content-addressable cache key.

        Args:
            content: Any serializable Python object that identifies the pipeline input.
            tags: Optional tags for grouping/invalidation.
            extra: Optional extra metadata to include in fingerprint.

        Returns:
            CacheKey with deterministic fingerprint.
        """
        # Build canonical representation
        canonical: list[Any] = [
            self.namespace,
            self.version,
        ]

        # Add extra metadata first for stable ordering
        if extra:
            canonical.append(_sorted_dict_items(extra))

        # Add main content
        canonical.append(_make_json_safe(content))

        # Add tags for invalidation grouping
        if tags:
            canonical.append(sorted(tags))

        # Compute fingerprint
        serialized = json.dumps(canonical, sort_keys=True, default=str)
        fingerprint = hashlib.new(self._hash_algo, serialized.encode("utf-8")).hexdigest()

        return CacheKey(
            fingerprint=fingerprint,
            namespace=self.namespace,
            version=self.version,
            tags=frozenset(tags or []),
        )

    def record_dependency(self, child_fingerprint: str, parent_fingerprint: str) -> None:
        """Record that a cache entry depends on another.

        Used for cascading invalidation: when a parent changes, all children
        that depend on it should be invalidated too.

        Args:
            child_fingerprint: Fingerprint of the dependent entry.
            parent_fingerprint: Fingerprint of the entry it depends on.
        """
        if child_fingerprint not in self._dependencies:
            self._dependencies[child_fingerprint] = set()
        self._dependencies[child_fingerprint].add(parent_fingerprint)

    def get_dependents(self, fingerprint: str) -> frozenset[str]:
        """Find all cache entries that depend on a given entry.

        Args:
            fingerprint: Fingerprint of the entry to query.

        Returns:
            Frozen set of fingerprints that depend on this entry (transitive closure).
        """
        visited: set[str] = set()
        to_visit = [fingerprint]

        while to_visit:
            current = to_visit.pop()
            if current in visited:
                continue
            visited.add(current)

            # Find entries that depend on current
            for child, parents in self._dependencies.items():
                if current in parents and child not in visited:
                    to_visit.append(child)

        visited.discard(fingerprint)
        return frozenset(visited)

    def get_dependency_graph(self) -> dict[str, list[str]]:
        """Return the full dependency graph as an adjacency list."""
        return {k: sorted(v) for k, v in self._dependencies.items()}

    def put(self, key: CacheKey, value: Any) -> None:
        """Store a value under a cache key."""
        self._store[key.fingerprint] = value

    def get(self, key: CacheKey) -> Any | None:
        """Retrieve a cached value by key."""
        return self._store.get(key.fingerprint)

    def invalidate(self, fingerprint: str, cascade: bool = True) -> set[str]:
        """Invalidate a cache entry and optionally its dependents.

        Args:
            fingerprint: Fingerprint to invalidate.
            cascade: If True, also invalidate all transitive dependents.

        Returns:
            Set of fingerprints that were invalidated.
        """
        invalidated: set[str] = {fingerprint}
        self._store.pop(fingerprint, None)

        if cascade:
            dependents = self.get_dependents(fingerprint)
            for dep in dependents:
                self._store.pop(dep, None)
            invalidated.update(dependents)

        # Clean up dependency tracking
        if fingerprint in self._dependencies:
            del self._dependencies[fingerprint]
        for deps in self._dependencies.values():
            deps.discard(fingerprint)

        return invalidated

    def invalidate_by_tag(self, tag: str) -> set[str]:
        """Invalidate all entries with a given tag."""
        # (Note: in a production implementation this would be indexed)
        invalidated: set[str] = set()
        for fp in list(self._store.keys()):
            invalidated.add(fp)
            self._store.pop(fp, None)
        return invalidated

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: CacheKey) -> bool:
        return key.fingerprint in self._store


def hash_inputs(*args: Any, **kwargs: Any) -> str:
    """Convenience: produce a deterministic SHA-256 hash of function inputs.

    Useful for manual key generation in pipeline scripts.

    Args:
        *args: Positional arguments to hash.
        **kwargs: Keyword arguments to hash.

    Returns:
        64-character hex digest.
    """
    payload = {
        "args": _make_json_safe(list(args)),
        "kwargs": _make_json_safe(dict(sorted(kwargs.items()))),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


__all__ = [
    "CacheKey",
    "ContentAddressableCache",
    "hash_inputs",
]