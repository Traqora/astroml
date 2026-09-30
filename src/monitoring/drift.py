def monitor_feature_drift(reference_dist, current_dist):
    # Fix: Add drift monitoring for feature distributions
    drift_score = abs(reference_dist.mean() - current_dist.mean())
    if drift_score > 0.05:
        print("Warning: Feature drift detected!")
    return drift_score
