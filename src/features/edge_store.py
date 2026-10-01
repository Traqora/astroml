def add_edge_derived_features(store):
    # Fix: Add edge-derived features to the feature store
    store.register_feature("edge_latency_ms")
    store.register_feature("edge_bandwidth_kbps")
