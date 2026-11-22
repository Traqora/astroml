import polars as pl
import numpy as np

def log_feature_duration(name, started, error=None):
    import time
    from src.utils.logger import log_structured
    log_structured("feature_computation", {"name": name, "duration_ms": round((time.perf_counter() - started) * 1000, 2), "status": "failed" if error else "ok"})

def compute_features_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Vectorize feature computation with Polars/NumPy
    return df.with_columns([
        (pl.col("raw_value") * np.pi).alias("transformed_value")
    ])

def handle_missing_features(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Add graceful missing/NaN feature handling
    return df.fill_null(strategy="forward").fill_nan(0.0)
