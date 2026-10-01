.. _graph-construction:

Graph construction architecture
===============================

.. contents:: On this page
   :local:
   :depth: 2

Who this is for
---------------

This is the maintainers' overview of how a table of Stellar transactions
becomes the sequence of graphs a model trains on. It is a companion to
``docs/graph-batch-processing.md``, which covers *tuning* the batch layer; this page
covers *what the layers are and why they are split where they are*.

Read it before changing anything under ``astroml/features/graph/``, and before
adding a new node or edge attribute -- a change that looks local to one module
usually invalidates an ordering assumption in another.

The one-sentence model
----------------------

A **dynamic graph** here means: the same entity set observed through a window
that slides forward in time, producing an ordered series of independent
snapshots -- one graph per window, in chronological order, each reproducible
from the database alone.

Everything else on this page follows from that. Because a snapshot is a pure
function of ``(window start, window end, edge filter)``, snapshots can be built
out of order, in parallel, and re-fetched years later -- and because the
*encoding* of a snapshot into tensors is where two runs can silently disagree,
that step is deliberately separated from the slicing.

Layer map
---------

::

                      PostgreSQL                       (ingestion writes this)
                         |
        normalized_transactions: sender, receiver, asset, amount, timestamp
                         |
                         v
   +-------------------------------------------+  snapshot.py
   |  WINDOW SLICING                           |  "which edges are in window i"
   |  iter_db_snapshots()    -> SnapshotWindow |
   |  iter_db_snapshot_edges() -> lazy edges   |
   |  parallel_build_snapshots() -> joblib     |
   |  window_snapshot(edges, start, end)       |  in-memory, bisect over ts
   +-------------------------------------------+
                         |  list[Edge] per window: Edge(src, dst, timestamp)
                         v
   +-------------------------------------------+  weights.py       (issue #731)
   |  OPTIONAL COLLAPSE                        |  build_weighted_snapshot()
   +-------------------------------------------+
                         |
                         v
   +-------------------------------------------+  edge_types.py    (issue #733)
   |  VOCABULARY + TENSOR ENCODING             |  build_typed_edge_index()
   +-------------------------------------------+   -> edge_index, edge_type, num_nodes
                         |                          (RGCNConv / HeteroConv ready)
                         v
   +-------------------------------------------+  neighborhood.py  (issue #735)
   |  FEATURES + STATISTICS                    |  precompute_neighborhood_features()
   |                                           |  statistics.py    (issue #740)
   +-------------------------------------------+   -> compute_snapshot_stats()
                         |
                         v
                      training loop  (subgraph.py #734 samples connected
                                      mini-batches from one snapshot)

   Cross-cutting: dsl.py (#739) wraps the whole column in a validated YAML
   document; batch_processor.py and dask_builder.py are alternate execution
   strategies for the slicing stage; memory_profile.py traces all of it.

Note that ``astroml/features/graph/__init__.py`` is intentionally empty. There
is no facade: callers import the specific submodule they need, so importing the
statistics helper never drags in the Dask or torch paths.

From relational rows to nodes and edges
---------------------------------------

The mapping is direct, and it is the whole reason a node id is a string:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Stellar thing
     - Becomes
     - Where
   * - Account (``G…`` address)
     - **Node**, keyed by the address string itself
     - ``Edge.src`` / ``Edge.dst``
   * - Payment from one account to another
     - **Directed edge** ``src -> dst``
     - ``NormalizedTransaction.sender`` / ``.receiver``
   * - Ledger close time
     - ``Edge.timestamp``, epoch **seconds** as an ``int``
     - ``NormalizedTransaction.timestamp``
   * - Asset (native XLM, ``USDC:G…``, …)
     - **Edge type**, not a node
     - ``NormalizedTransaction.asset`` -> ``EdgeTypeVocabulary``
   * - Amount
     - **Edge weight**, not a node
     - ``NormalizedTransaction.amount`` -> ``WeightSpec``
   * - Memo, operation type, transaction hash
     - Not represented in the topology
     - dropped at slicing time

Two filters are applied in the SQL of every snapshot query, and they define
what the graph *is*:

.. code-block:: sql

   receiver IS NOT NULL AND sender != receiver

- ``receiver IS NOT NULL`` -- a transaction with no counterparty (a contract
  call, a mnemonic-setting operation) is not a payment, so it is not an edge.
- ``sender != receiver`` -- self-payments are excluded rather than kept as
  self-loops, because a self-loop makes degree-based features lie about a
  node's connectivity.

If you want either class back, that is a change to the *filter*, and every
existing snapshot changes shape. It is not a flag on the builder today.

Asset-as-edge-type rather than asset-as-node is the other load-bearing choice:
it keeps the node set = the account set stable across windows, so a model's
output layer does not resize between snapshots.

Rolling time windows
--------------------

A window is described by two durations: its **size** and the **step** by which
it slides. From :func:`~astroml.features.graph.snapshot.iter_db_snapshots`:

- ``step == window`` (the default) gives **non-overlapping** tiling: every edge
  lands in exactly one snapshot.
- ``step < window`` gives **overlapping / rolling** windows: an edge appears in
  ``ceil(window / step)`` consecutive snapshots. This is what you want for a
  model that must not see a gap between observations -- at the cost of training
  on correlated windows.

Both are parsed from strings like ``'7d'``, ``'24h'``, ``'3600s'`` by
``_parse_window_size``, which supports only ``d``, ``h`` and ``s``. A value of
zero or less is rejected: it used to be accepted, and produced an
``iter_db_snapshots`` loop that never advanced (issue #991).

The bounds are **inclusive on both ends** (``>= start AND <= end``), and the
last window is clamped:

.. code-block:: text

   t0                                       t_now = now(UTC)
   |---- window 0 ----|
             |---- window 1 ----|
                       ...
                                 |-- window n --|   <- end clamped to t_now

   window_start advances by `step`, and iteration stops at window_start >= t_now

So the final window can be *shorter* than requested. Nothing pads it, and
:func:`~astroml.features.graph.snapshot.snapshot_last_n_days` documents the
same inclusive convention for the ad-hoc case.

Four ways to get the same windows
---------------------------------

These are alternative entry points, not a pipeline -- each yields windowed
edge sets and hands off to the encoding stage above.

``iter_db_snapshots(window, t0, t_now, step, session, chunk_size, workers)``
   The default. Materialises ``edges`` and ``nodes`` for one
   :class:`SnapshotWindow` at a time, streaming rows with SQLAlchemy
   ``yield_per(chunk_size)`` (default 100 000). With ``workers > 1`` it
   prefetches windows on a thread pool and **re-sorts them back into index
   order before yielding**, so parallelism never changes the sequence. The
   threaded path only applies when you did *not* pass your own ``session`` --
   a session is not safe to share across those threads.

``iter_db_snapshot_edges(...)`` -- issue #199
   Yields ``(SnapshotMeta, Iterator[Edge])`` instead of a materialised window,
   with a smaller default ``chunk_size`` (5 000). Peak memory per window is
   bounded by that chunk regardless of how many edges the window holds. The
   trade-off: the edge iterator must be drained or discarded before you take
   the next pair, because it reuses the underlying SQL result. Use this when a
   window could plausibly OOM the box.

``parallel_build_snapshots(..., n_jobs=-1, backend='loky', batch_size=None)``
   joblib across whole windows; returns a **list**, in chronological order.
   Falls back to sequential ``iter_db_snapshots`` when joblib is not installed.
   Each worker opens its own session, which is why it ignores your ``session``.
   Set ``batch_size`` to cap how many finished windows are alive at once.

``window_snapshot(edges, start_ts, end_ts, presorted=True)``
   No database at all: an in-memory slice over a caller-supplied edge list,
   using two ``bisect`` calls on the timestamp array, so ``O(log N + K)``.
   ``presorted=True`` is a *promise*, not a check -- it skips the sort. Pass it
   on unsorted input and you get a wrong answer, not an error.

How snapshots are stored
------------------------

There is no snapshot table, and no snapshot files on disk. **The database is
the store of record**, and a snapshot is re-derivable at any time from
``normalized_transactions`` plus its window parameters. This is a deliberate
design position, and the reason the whole layer has no save/load API:

- A snapshot is reproducible only if it is never stored. Persisting one invites
  training on a graph whose source rows have since been corrected.
- The only durable identifiers are the window bounds and, for the declarative
  path, :meth:`GraphSpec.fingerprint <astroml.features.graph.dsl.GraphSpec.fingerprint>`
  -- a stable hash recorded *with* the run so you know which spec produced it.
- Per-snapshot **statistics** are persisted as a JSON report by
  :func:`~astroml.features.graph.statistics.compute_snapshot_stats` (issue
  #740). That report is deterministic and deliberately excludes a
  ``generated_at`` stamp, so two runs over one snapshot diff cleanly. It is a
  record *about* the snapshot, not the snapshot.

What *is* cached is the memo layer in Redis, via
``@cached_graph_snapshot(ttl_seconds=1800)`` on ``window_snapshot`` and
``snapshot_last_n_days``. Two properties of it matter to correctness:

.. warning::

   **The cache key is content-blind for sequence arguments.**
   ``RedisCache._hash_key`` reduces a ``list``, ``tuple`` or ``set`` to the
   string ``list:<length>``. So ``window_snapshot(edges_a, t0, t1)`` and
   ``window_snapshot(edges_b, t0, t1)`` collide whenever ``edges_a`` and
   ``edges_b`` have the same length, and whichever ran first wins for 30
   minutes. Keying on the actual edges (or passing ``key_func``) is required
   before this decorator is safe on any path where two distinct edge sets of
   equal size can be queried over the same window.

   **It is a cache, not a store, and it degrades silently.** Every Redis
   error is caught, logged as a warning and reported as a miss. With no Redis
   reachable the builder recomputes and everything still works. Nothing in this
   layer will tell you it has stopped being fast -- watch the cache stats, not
   the logs.

Ordering and determinism
------------------------

The invariant across the layer: **the same edge set always produces the same
output, independent of arrival order.** Each stage enforces its own share, and
this is the most common thing a change breaks.

- Slicing orders SQL by ``timestamp``; the in-memory helpers accept an
  unsorted list only when told to sort it.
- ``build_weighted_snapshot`` sorts before accumulating, which is what makes
  ``sum``/``mean`` order-independent.
- ``build_typed_edge_index`` emits **nodes sorted** and edges in canonical
  order, so ``node i`` means the same account in two runs.
- ``precompute_neighborhood_features`` iterates a sorted adjacency.
- The parallel paths re-sort results by window index before yielding.

The subtle one is the edge-type vocabulary. Ids are assigned in **sorted order
of the type key**, never insertion order:

.. code-block:: python

   # Two shards, processed in either order: "USDC" is relation 1 in both.
   vocab = EdgeTypeVocabulary.build(edges, spec=EdgeTypeSpec(fields=("asset",)))

Had ids come from "next id for each newly seen type", a model trained on
Monday's shard would read Tuesday's ``edge_type=3`` as a different relation --
with no error, only a worse model. Persist and reuse the training vocabulary
by passing it as ``vocabulary=`` when encoding evaluation data.

A type absent from that vocabulary raises :class:`UnknownEdgeTypeError` by
default. ``allow_unknown=True`` instead routes it to a reserved bucket at id
``0`` (``UNKNOWN_TYPE``), which is the choice for live serving, where refusing
the request is not an option.

Declarative entry point
-----------------------

:mod:`~astroml.features.graph.dsl` (issue #739) turns the above into a document
so a run can be reproduced without reading last month's script:

.. code-block:: yaml

   version: 1
   name: weekly-payments
   window:
     size: 7d
     step: 1d          # step < size => rolling, overlapping windows
   edges:
     types: [payment, path_payment]
     min_amount: 10.0
     exclude_self_loops: true
     directed: true
   features:
     max_hops: 2
     neighbour_aggregates: true
   nodes:
     attributes: [degree, total_sent, hop2_size]

``GraphSpec.from_yaml`` validates strictly -- unknown keys are rejected, and
every problem in the document is reported at once. A spec that quietly ignored
``mim_amount`` would be worse than one that refused to load.

A note on the amount filter
---------------------------

The ``min_amount`` / ``exclude_self_loops`` / ``types`` filters live in the
declarative spec. The raw ``iter_db_snapshots`` path applies only the two SQL
filters shown above, so a spec built through ``build_from_spec`` and a hand
rolled ``iter_db_snapshots(window="7d")`` call over the same database are
**not** the same graph unless the spec is left at its defaults. When a
snapshot and a spec disagree, check here first.

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - If you want to
     - Look at
   * - Choose or tune a batch size
     - ``docs/graph-batch-processing.md``
   * - Sample a mini-batch that fits a GPU
     - ``astroml/features/graph/subgraph.py`` (issue #734)
   * - Add a node feature
     - ``astroml/features/graph/neighborhood.py`` (issue #735)
   * - Add or rename an edge type
     - ``astroml/features/graph/edge_types.py`` (issue #733)
   * - Change how weights combine
     - ``astroml/features/graph/weights.py`` (issue #731)
   * - Diff two snapshots to explain a regression
     - ``astroml/features/graph/statistics.py`` (issue #740)
   * - Scale past a single machine
     - ``astroml/features/graph/dask_builder.py``
   * - Trace where memory goes
     - ``astroml/features/graph/memory_profile.py`` and ``cli.py`` (issue #546)

Run it yourself
---------------

The two stages below run with no database. They are separate calls on purpose:
:class:`~astroml.features.graph.snapshot.Edge` carries only ``src``, ``dst`` and
``timestamp``, while the type key is read from an ``asset`` field, so typing
works on the richer source records rather than on a sliced ``Edge`` list.

Slice a window -- bounds are inclusive on both sides, and a window that
contains no edges is empty rather than an error:

.. code-block:: python

   from astroml.features.graph.snapshot import Edge, window_snapshot

   edges = [
       Edge(src="GALICE", dst="GBOB", timestamp=1_700_000_000),
       Edge(src="GBOB", dst="GCAROL", timestamp=1_700_000_600),
       Edge(src="GCAROL", dst="GALICE", timestamp=1_700_010_000),
   ]
   nodes, window = window_snapshot(edges, 1_700_000_000, 1_700_001_000)

   print(len(window))  # 2 -- the 10_000-second edge is outside the window
   print(sorted(nodes))  # ['GALICE', 'GBOB', 'GCAROL']

Build a typed ``edge_index`` from records that do carry their asset. Mappings
are accepted directly, so no adapter is needed for rows that came from the
ledger:

.. code-block:: python

   from astroml.features.graph.edge_types import EdgeTypeSpec, build_typed_edge_index

   typed = build_typed_edge_index(
       [
           {"src": "GALICE", "dst": "GBOB", "asset": "USDC"},
           {"src": "GBOB", "dst": "GCAROL", "asset": "XLM"},
       ],
       spec=EdgeTypeSpec(fields=("asset",)),
   )

   print(typed.nodes)          # ('GALICE', 'GBOB', 'GCAROL') -- sorted, not seen-order
   print(typed.edge_index)     # ((0, 1), (1, 2)) -- rows are node indices
   print(typed.edge_type)      # (0, 1) -- USDC and XLM are distinct relations
   print(typed.num_relations)  # 2

``to_tensors()`` converts the pair for PyG, and passing a saved
``vocabulary=`` re-encodes a later window against the same relation ids.

For the database-backed version, the same import plus
``iter_db_snapshots(window="7d", step="1d")`` in a ``for`` loop yields the
chronological series; nothing needs to be materialised in advance.
