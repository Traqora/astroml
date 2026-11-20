import polars as pl
import numpy as np

def compute_features_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Vectorize feature computation with Polars/NumPy
    import time
    from src.utils.logger import log_structured
    started = time.perf_counter()
    try:
        result = df.with_columns([(pl.col("raw_value") * np.pi).alias("transformed_value")])
        log_structured("feature_compute", {"stage": "features", "status": "ok", "duration_ms": (time.perf_counter() - started) * 1000})
        return result
    except Exception as exc:
        log_structured("feature_compute", {"stage": "features", "status": "failed", "error": str(exc)})
        raise

def handle_missing_features(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Add graceful missing/NaN feature handling
    return df.fill_null(strategy="forward").fill_nan(0.0)
