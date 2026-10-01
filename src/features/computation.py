import polars as pl
import numpy as np

def compute_features_vectorized(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Vectorize feature computation with Polars/NumPy
    return df.with_columns([
        (pl.col("raw_value") * np.pi).alias("transformed_value")
    ])

def handle_missing_features(df: pl.DataFrame) -> pl.DataFrame:
    # Fix: Add graceful missing/NaN feature handling
    return df.fill_null(strategy="forward").fill_nan(0.0)
