from __future__ import annotations

import pandas as pd
from astroml.features.node_features import compute_node_features


def test_compute_node_features_basic():
  edges = [
    { 'src': 'A', 'dst': 'B', 'amount': 10, 'timestamp': 100 },
    { 'src': 'A', 'dst': 'C', 'amount': 5,  'timestamp': 110 },
    { 'src': 'B', 'dst': 'A', 'amount': 2,  'timestamp': 120 },
    { 'src': 'C', 'dst': 'A', 'amount': 3,  'timestamp': 130 },
  ]

  feats = compute_node_features(edges)

  # Degrees
  assert feats.loc['A','out_degree'] == 2
  assert feats.loc['A','in_degree'] == 2
  assert feats.loc['B','out_degree'] == 1
  assert feats.loc['B','in_degree'] == 1
  assert feats.loc['C','out_degree'] == 1
  assert feats.loc['C','in_degree'] == 1

  # Volumes
  assert feats.loc['A','total_sent'] == 15
  assert feats.loc['A','total_received'] == 5
  assert feats.loc['B','total_sent'] == 2
  assert feats.loc['B','total_received'] == 10
  assert feats.loc['C','total_sent'] == 3
  assert feats.loc['C','total_received'] == 5

  # Account age with default ref_time = max ts in edges (130)
  # First seen per node (min ts among incident edges): A:100, B:100, C:110
  assert feats.loc['A','account_age'] == 30
  assert feats.loc['B','account_age'] == 30
  assert feats.loc['C','account_age'] == 20


def test_compute_node_features_with_provided_first_seen_and_ref_time():
  edges = [
    { 'src': 'A', 'dst': 'B', 'amount': 10, 'timestamp': 100 },
  ]
  first_seen = { 'A': 90, 'B': 95, 'D': 50 }  # D has no edges
  feats = compute_node_features(edges, nodes_first_seen=first_seen, ref_time=200)

  # Node D should be included with zeros for degree/volume
  assert 'D' in feats.index
  assert feats.loc['D','in_degree'] == 0
  assert feats.loc['D','out_degree'] == 0
  assert feats.loc['D','total_received'] == 0
  assert feats.loc['D','total_sent'] == 0

  # Account age uses provided first_seen and given ref_time
  assert feats.loc['A','account_age'] == 110  # 200 - 90
  assert feats.loc['B','account_age'] == 105  # 200 - 95
  assert feats.loc['D','account_age'] == 150  # 200 - 50
