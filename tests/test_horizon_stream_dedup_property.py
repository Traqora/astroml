"""Property-based tests for HorizonStreamingClient's replay de-duplication.

Issue #836: Edge dedup property test.

SSE-based delivery (see `HorizonStreamingClient`'s class docstring) is
at-least-once: a reconnect can replay transactions already delivered. The
client's own de-duplication (`_already_delivered`/`_remember`, backed by a
bounded `OrderedDict`) is what turns that into effectively-once delivery
for any `paging_token` still within `dedupe_capacity`. These tests check
that invariant directly against the de-dup machinery, for arbitrary
sequences of tokens, rather than against one or two hand-picked cases.
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from astroml.ingestion.horizon_stream import HorizonStreamingClient

# Keep examples small and the capacity bounded so shrinking stays fast and
# the "never exceeds capacity" property is actually exercised, not just
# trivially true because nothing ever filled the window.
_TOKEN_SEQUENCES = st.lists(st.text(min_size=1, max_size=8), min_size=0, max_size=60)
_CAPACITIES = st.integers(min_value=1, max_value=20)


def _feed(client: HorizonStreamingClient, tokens: list[str]) -> list[bool]:
    """Replay `tokens` through the dedup machinery, as `_handle_payload`
    would: check-then-remember, in order. Returns, per token, whether it
    was judged a duplicate."""
    results = []
    for token in tokens:
        is_duplicate = client._already_delivered(token)
        results.append(is_duplicate)
        if not is_duplicate:
            client._remember(token)
    return results


class TestHorizonDedupProperties:
    @given(tokens=_TOKEN_SEQUENCES, capacity=_CAPACITIES)
    @settings(max_examples=200, deadline=None)
    def test_first_occurrence_of_any_token_is_never_a_duplicate(
        self, tokens: list[str], capacity: int
    ) -> None:
        client = HorizonStreamingClient(dedupe_capacity=capacity)
        is_duplicate = _feed(client, tokens)

        seen_before: set[str] = set()
        for token, was_duplicate in zip(tokens, is_duplicate):
            if token not in seen_before:
                assert not was_duplicate
            seen_before.add(token)

    @given(tokens=_TOKEN_SEQUENCES, capacity=_CAPACITIES)
    @settings(max_examples=200, deadline=None)
    def test_a_token_repeated_within_capacity_is_always_flagged_duplicate(
        self, tokens: list[str], capacity: int
    ) -> None:
        """A duplicate hit does not call `_remember` (matching the real
        `_handle_payload` flow: only a newly-delivered token's position is
        touched), so it does not refresh that token's place in the LRU
        window either. The right bound to check against is therefore the
        number of *other* tokens actually inserted into the window (i.e.
        first-time occurrences) since this token's own last insertion, not
        since its last occurrence in the raw input (which may include
        duplicate hits that insert nothing)."""
        client = HorizonStreamingClient(dedupe_capacity=capacity)
        is_duplicate = _feed(client, tokens)

        last_inserted_at: dict[str, int] = {}
        inserted_tokens_in_order: list[str] = []
        for i, token in enumerate(tokens):
            if token in last_inserted_at:
                distinct_insertions_since = len(
                    set(inserted_tokens_in_order[last_inserted_at[token] + 1 :])
                )
                if distinct_insertions_since < capacity:
                    assert is_duplicate[i], (
                        f"token {token!r} at index {i} should have been flagged "
                        f"duplicate: only {distinct_insertions_since} other distinct "
                        f"tokens inserted since its last insertion, capacity={capacity}"
                    )
            if not is_duplicate[i]:
                last_inserted_at[token] = len(inserted_tokens_in_order)
                inserted_tokens_in_order.append(token)

    @given(tokens=_TOKEN_SEQUENCES, capacity=_CAPACITIES)
    @settings(max_examples=200, deadline=None)
    def test_remembered_set_never_exceeds_capacity(self, tokens: list[str], capacity: int) -> None:
        client = HorizonStreamingClient(dedupe_capacity=capacity)
        for token in tokens:
            if not client._already_delivered(token):
                client._remember(token)
            assert len(client._seen) <= capacity

    @given(tokens=_TOKEN_SEQUENCES, capacity=_CAPACITIES)
    @settings(max_examples=200, deadline=None)
    def test_remembering_is_idempotent(self, tokens: list[str], capacity: int) -> None:
        """Remembering the same token twice in a row must not grow the
        window by more than one entry, and must not change whether other
        tokens are still remembered (beyond normal LRU eviction order)."""
        client = HorizonStreamingClient(dedupe_capacity=capacity)
        for token in tokens:
            client._remember(token)
            size_after_first = len(client._seen)
            client._remember(token)
            size_after_second = len(client._seen)
            assert size_after_second == size_after_first

    def test_eviction_is_oldest_first(self) -> None:
        """Concrete edge case: filling exactly past capacity evicts the
        single oldest token, not an arbitrary one."""
        client = HorizonStreamingClient(dedupe_capacity=3)
        for token in ("a", "b", "c"):
            client._remember(token)
        assert client._already_delivered("a")

        client._remember("d")
        assert not client._already_delivered("a")
        assert client._already_delivered("b")
        assert client._already_delivered("c")
        assert client._already_delivered("d")

    def test_re_delivery_of_an_evicted_token_is_not_treated_as_duplicate(self) -> None:
        """Concrete edge case: once evicted, a token is indistinguishable
        from one never seen before — this is the documented trade-off of a
        bounded dedup window, not a bug, but it must behave consistently."""
        client = HorizonStreamingClient(dedupe_capacity=1)
        client._remember("a")
        assert client._already_delivered("a")

        client._remember("b")
        assert not client._already_delivered("a")

        # Re-delivering "a" now looks like a brand-new token.
        assert not client._already_delivered("a")
        client._remember("a")
        assert client._already_delivered("a")

    def test_empty_token_sequence_leaves_an_empty_window(self) -> None:
        client = HorizonStreamingClient(dedupe_capacity=5)
        assert len(client._seen) == 0
        assert not client._already_delivered("anything")

    @given(capacity=_CAPACITIES)
    @settings(max_examples=50, deadline=None)
    def test_exactly_capacity_distinct_tokens_all_survive(self, capacity: int) -> None:
        client = HorizonStreamingClient(dedupe_capacity=capacity)
        tokens = [str(i) for i in range(capacity)]
        for token in tokens:
            client._remember(token)
        for token in tokens:
            assert client._already_delivered(token)
