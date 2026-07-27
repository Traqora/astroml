"""Feature transformation utilities for the Feature Store.

This module provides various transformers for preprocessing and engineering
features before they are stored or used in machine learning models.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple
from enum import Enum

import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    PowerTransformer,
    LabelEncoder,
    OneHotEncoder,
)
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)


class TransformationType(Enum):
    """Types of feature transformations."""
    STANDARD_SCALER = "standard_scaler"
    MIN_MAX_SCALER = "min_max_scaler"
    ROBUST_SCALER = "robust_scaler"
    QUANTILE_TRANSFORMER = "quantile_transformer"
    POWER_TRANSFORMER = "power_transformer"
    LABEL_ENCODER = "label_encoder"
    ONE_HOT_ENCODER = "one_hot_encoder"
    LOG_TRANSFORM = "log_transform"
    BOX_COX = "box_cox"
    YEO_JOHNSON = "yeo_johnson"
    BUCKETIZE = "bucketize"
    CUSTOM = "custom"


@dataclass
class TransformationConfig:
    """Configuration for feature transformation.
    
    Attributes:
        transformation_type: Type of transformation
        parameters: Transformation parameters
        input_columns: Input column names
        output_columns: Output column names
        fitted_params: Fitted transformation parameters
    """
    
    transformation_type: TransformationType
    parameters: Dict[str, Any]
    input_columns: List[str]
    output_columns: List[str]
    fitted_params: Dict[str, Any] = None
    
    def __post_init__(self) -> None:
        if self.fitted_params is None:
            self.fitted_params = {}


class LogTransformer(BaseEstimator, TransformerMixin):
    """Custom log transformer with handling of zeros and negative values."""
    
    def __init__(self, offset: float = 1.0, handle_negative: str = "error"):
        """Initialize log transformer.
        
        Args:
            offset: Offset to add before log transformation
            handle_negative: How to handle negative values ('error', 'abs', 'clip')
        """
        self.offset = offset
        self.handle_negative = handle_negative
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> LogTransformer:
        """Fit transformer (no-op for log transform)."""
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation."""
        X_transformed = X.copy()
        
        for col in X.columns:
            values = X[col].astype(float)
            
            # Handle negative values
            if self.handle_negative == "error" and (values < 0).any():
                raise ValueError(f"Negative values found in column {col}")
            elif self.handle_negative == "abs":
                values = values.abs()
            elif self.handle_negative == "clip":
                values = values.clip(lower=0)
            
            # Apply log transformation
            X_transformed[col] = np.log(values + self.offset)
        
        return X_transformed


class Bucketizer(BaseEstimator, TransformerMixin):
    """Custom bucketizer for continuous features."""
    
    def __init__(self, n_bins: int = 10, strategy: str = "uniform", labels: Optional[List[str]] = None):
        """Initialize bucketizer.
        
        Args:
            n_bins: Number of bins
            strategy: Binning strategy ('uniform', 'quantile', 'kmeans')
            labels: Optional bin labels
        """
        self.n_bins = n_bins
        self.strategy = strategy
        self.labels = labels
        self.bin_edges_: Dict[str, np.ndarray] = {}
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Bucketizer:
        """Fit bucketizer."""
        for col in X.columns:
            if self.strategy == "uniform":
                _, bin_edges = pd.cut(X[col], bins=self.n_bins, retbins=True)
            elif self.strategy == "quantile":
                _, bin_edges = pd.qcut(X[col], q=self.n_bins, retbins=True, duplicates='drop')
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            self.bin_edges_[col] = bin_edges
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply bucketization."""
        X_transformed = X.copy()
        
        for col in X.columns:
            if col in self.bin_edges_:
                if self.labels:
                    X_transformed[col] = pd.cut(
                        X[col], 
                        bins=self.bin_edges_[col], 
                        labels=self.labels[:len(self.bin_edges_[col])-1],
                        include_lowest=True
                    )
                else:
                    X_transformed[col] = pd.cut(
                        X[col], 
                        bins=self.bin_edges_[col], 
                        include_lowest=True
                    )
        
        return X_transformed


class FeatureTransformer:
    """Main feature transformer class.
    
    Provides a unified interface for applying various transformations
    to features with support for fitting, transforming, and persistence.
    """
    
    def __init__(self):
        """Initialize feature transformer."""
        self.transformers: Dict[str, BaseEstimator] = {}
        self.configs: Dict[str, TransformationConfig] = {}
        self._fitted = False
    
    def add_transformation(
        self,
        name: str,
        transformation_type: TransformationType,
        input_columns: List[str],
        output_columns: Optional[List[str]] = None,
        **parameters: Any,
    ) -> None:
        """Add a transformation to the pipeline.
        
        Args:
            name: Transformation name
            transformation_type: Type of transformation
            input_columns: Input column names
            output_columns: Output column names (defaults to input_columns)
            **parameters: Transformation parameters
        """
        if output_columns is None:
            output_columns = input_columns
        
        config = TransformationConfig(
            transformation_type=transformation_type,
            parameters=parameters,
            input_columns=input_columns,
            output_columns=output_columns,
        )
        
        self.configs[name] = config
        
        # Create transformer instance
        transformer = self._create_transformer(transformation_type, **parameters)
        self.transformers[name] = transformer
        
        logger.info(f"Added transformation '{name}' of type {transformation_type.value}")
    
    def _create_transformer(self, transformation_type: TransformationType, **parameters: Any) -> BaseEstimator:
        """Create transformer instance based on type."""
        if transformation_type == TransformationType.STANDARD_SCALER:
            return StandardScaler(**parameters)
        elif transformation_type == TransformationType.MIN_MAX_SCALER:
            return MinMaxScaler(**parameters)
        elif transformation_type == TransformationType.ROBUST_SCALER:
            return RobustScaler(**parameters)
        elif transformation_type == TransformationType.QUANTILE_TRANSFORMER:
            return QuantileTransformer(**parameters)
        elif transformation_type == TransformationType.POWER_TRANSFORMER:
            return PowerTransformer(**parameters)
        elif transformation_type == TransformationType.LABEL_ENCODER:
            return LabelEncoder()
        elif transformation_type == TransformationType.ONE_HOT_ENCODER:
            return OneHotEncoder(**parameters)
        elif transformation_type == TransformationType.LOG_TRANSFORM:
            return LogTransformer(**parameters)
        elif transformation_type == TransformationType.BUCKETIZE:
            return Bucketizer(**parameters)
        else:
            raise ValueError(f"Unknown transformation type: {transformation_type}")
    
    def fit(self, data: pd.DataFrame) -> FeatureTransformer:
        """Fit all transformations.
        
        Args:
            data: Input data
            
        Returns:
            Self for method chaining
        """
        for name, transformer in self.transformers.items():
            config = self.configs[name]
            input_data = data[config.input_columns]
            
            logger.info(f"Fitting transformation '{name}'")
            transformer.fit(input_data)
        
        self._fitted = True
        logger.info("All transformations fitted")
        return self
    
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all transformations.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        if not self._fitted:
            raise RuntimeError("Transformer must be fitted before transformation")
        
        result = data.copy()
        
        for name, transformer in self.transformers.items():
            config = self.configs[name]
            input_data = data[config.input_columns]
            
            logger.info(f"Applying transformation '{name}'")
            
            # Apply transformation
            if isinstance(transformer, (LabelEncoder, OneHotEncoder)):
                # Handle encoders differently
                if isinstance(transformer, LabelEncoder):
                    for i, col in enumerate(config.input_columns):
                        if len(config.input_columns) == 1:
                            transformed = transformer.transform(input_data.iloc[:, i])
                        else:
                            transformed = transformer.transform(input_data.iloc[:, i])
                        result[config.output_columns[i]] = transformed
                else:  # OneHotEncoder
                    transformed = transformer.transform(input_data)
                    # Create column names for one-hot encoded features
                    feature_names = []
                    for i, col in enumerate(config.input_columns):
                        if hasattr(transformer, 'categories_'):
                            categories = transformer.categories_[i]
                            for category in categories:
                                feature_names.append(f"{col}_{category}")
                    
                    transformed_df = pd.DataFrame(
                        transformed.toarray() if hasattr(transformed, 'toarray') else transformed,
                        columns=feature_names,
                        index=data.index
                    )
                    
                    # Remove original columns and add encoded columns
                    result = result.drop(columns=config.input_columns)
                    result = pd.concat([result, transformed_df], axis=1)
            else:
                # Handle other transformers
                transformed = transformer.transform(input_data)
                if isinstance(transformed, np.ndarray):
                    transformed_df = pd.DataFrame(
                        transformed,
                        columns=config.output_columns,
                        index=data.index
                    )
                    result[config.output_columns] = transformed_df
                else:
                    # DataFrame output
                    result[config.output_columns] = transformed
        
        return result
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Args:
            data: Input data
            
        Returns:
            Transformed data
        """
        return self.fit(data).transform(data)
    
    def get_config(self, name: str) -> Optional[TransformationConfig]:
        """Get transformation configuration.
        
        Args:
            name: Transformation name
            
        Returns:
            Transformation configuration if found
        """
        return self.configs.get(name)
    
    def list_transformations(self) -> List[str]:
        """List all transformation names."""
        return list(self.configs.keys())
    
    def remove_transformation(self, name: str) -> None:
        """Remove a transformation.
        
        Args:
            name: Transformation name
        """
        if name in self.configs:
            del self.configs[name]
        if name in self.transformers:
            del self.transformers[name]
        
        logger.info(f"Removed transformation '{name}'")
    
    def save(self, filepath: str) -> None:
        """Save transformer configuration and fitted parameters.
        
        Args:
            filepath: Path to save configuration
        """
        import pickle
        
        save_data = {
            "configs": self.configs,
            "transformers": self.transformers,
            "fitted": self._fitted,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved transformer to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> FeatureTransformer:
        """Load transformer from file.
        
        Args:
            filepath: Path to load configuration from
            
        Returns:
            Loaded transformer
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        transformer = cls()
        transformer.configs = save_data["configs"]
        transformer.transformers = save_data["transformers"]
        transformer._fitted = save_data["fitted"]
        
        logger.info(f"Loaded transformer from {filepath}")
        return transformer


class FeatureEngineering:
    """Advanced feature engineering utilities."""
    
    @staticmethod
    def create_interaction_features(
        data: pd.DataFrame,
        columns: List[str],
        interaction_type: str = "multiplication",
    ) -> pd.DataFrame:
        """Create interaction features between columns.
        
        Args:
            data: Input data
            columns: Columns to create interactions from
            interaction_type: Type of interaction ('multiplication', 'addition', 'subtraction')
            
        Returns:
            DataFrame with interaction features
        """
        result = data.copy()
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns[i+1:], i+1):
                if col1 not in data.columns or col2 not in data.columns:
                    continue
                
                if interaction_type == "multiplication":
                    result[f"{col1}_x_{col2}"] = data[col1] * data[col2]
                elif interaction_type == "addition":
                    result[f"{col1}_plus_{col2}"] = data[col1] + data[col2]
                elif interaction_type == "subtraction":
                    result[f"{col1}_minus_{col2}"] = data[col1] - data[col2]
                    result[f"{col2}_minus_{col1}"] = data[col2] - data[col1]
        
        return result
    
    @staticmethod
    def create_polynomial_features(
        data: pd.DataFrame,
        columns: List[str],
        degree: int = 2,
    ) -> pd.DataFrame:
        """Create polynomial features.
        
        Args:
            data: Input data
            columns: Columns to create polynomial features from
            degree: Polynomial degree
            
        Returns:
            DataFrame with polynomial features
        """
        from sklearn.preprocessing import PolynomialFeatures
        
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            poly = PolynomialFeatures(degree=degree, include_bias=False)
            poly_features = poly.fit_transform(data[[col]])
            
            feature_names = poly.get_feature_names_out([col])
            
            # Skip the original column (degree 1)
            for i, name in enumerate(feature_names):
                if name != col:
                    result[name] = poly_features[:, i]
        
        return result
    
    @staticmethod
    def create_rolling_features(
        data: pd.DataFrame,
        columns: List[str],
        window_sizes: List[int],
        functions: List[str] = ["mean", "std", "min", "max"],
    ) -> pd.DataFrame:
        """Create rolling window features.
        
        Args:
            data: Input data with datetime index
            columns: Columns to create rolling features from
            window_sizes: List of window sizes
            functions: List of aggregation functions
            
        Returns:
            DataFrame with rolling features
        """
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            for window in window_sizes:
                for func in functions:
                    feature_name = f"{col}_rolling_{window}_{func}"
                    
                    if func == "mean":
                        result[feature_name] = data[col].rolling(window=window).mean()
                    elif func == "std":
                        result[feature_name] = data[col].rolling(window=window).std()
                    elif func == "min":
                        result[feature_name] = data[col].rolling(window=window).min()
                    elif func == "max":
                        result[feature_name] = data[col].rolling(window=window).max()
                    elif func == "sum":
                        result[feature_name] = data[col].rolling(window=window).sum()
                    elif func == "median":
                        result[feature_name] = data[col].rolling(window=window).median()
        
        return result
    
    @staticmethod
    def create_lag_features(
        data: pd.DataFrame,
        columns: List[str],
        lags: List[int],
    ) -> pd.DataFrame:
        """Create lag features.
        
        Args:
            data: Input data with datetime index
            columns: Columns to create lag features from
            lags: List of lag periods
            
        Returns:
            DataFrame with lag features
        """
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            for lag in lags:
                feature_name = f"{col}_lag_{lag}"
                result[feature_name] = data[col].shift(lag)
        
        return result
    
    @staticmethod
    def create_time_features(
        data: pd.DataFrame,
        timestamp_column: str,
    ) -> pd.DataFrame:
        """Create time-based features from timestamp column.
        
        Args:
            data: Input data
            timestamp_column: Name of timestamp column
            
        Returns:
            DataFrame with time features
        """
        result = data.copy()
        
        if timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found")
        
        # Convert to datetime if needed
        timestamps = pd.to_datetime(data[timestamp_column])
        
        # Extract time components
        result["hour"] = timestamps.dt.hour
        result["day_of_week"] = timestamps.dt.dayofweek
        result["day_of_month"] = timestamps.dt.day
        result["month"] = timestamps.dt.month
        result["quarter"] = timestamps.dt.quarter
        result["year"] = timestamps.dt.year
        
        # Cyclical features
        result["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
        result["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
        result["day_sin"] = np.sin(2 * np.pi * timestamps.dt.dayofweek / 7)
        result["day_cos"] = np.cos(2 * np.pi * timestamps.dt.dayofweek / 7)
        result["month_sin"] = np.sin(2 * np.pi * timestamps.dt.month / 12)
        result["month_cos"] = np.cos(2 * np.pi * timestamps.dt.month / 12)
        
        # Weekend indicator
        result["is_weekend"] = timestamps.dt.dayofweek.isin([5, 6]).astype(int)
        
        return result
    
    @staticmethod
    def detect_outliers(
        data: pd.DataFrame,
        columns: List[str],
        method: str = "iqr",
        threshold: float = 1.5,
    ) -> pd.DataFrame:
        """Detect outliers in specified columns.
        
        Args:
            data: Input data
            columns: Columns to check for outliers
            method: Outlier detection method ('iqr', 'zscore', 'isolation_forest')
            threshold: Threshold for outlier detection
            
        Returns:
            DataFrame with outlier indicators
        """
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                continue
            
            outlier_col = f"{col}_outlier"
            
            if method == "iqr":
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                result[outlier_col] = ((data[col] < lower_bound) | (data[col] > upper_bound)).astype(int)
            
            elif method == "zscore":
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                result[outlier_col] = (z_scores > threshold).astype(int)
            
            elif method == "isolation_forest":
                from sklearn.ensemble import IsolationForest
                
                iso_forest = IsolationForest(contamination=0.1, random_state=42)
                outliers = iso_forest.fit_predict(data[[col]])
                result[outlier_col] = (outliers == -1).astype(int)
        
        return result


# Convenience functions

def create_feature_transformer() -> FeatureTransformer:
    """Create a new feature transformer instance.
    
    Returns:
        Feature transformer instance
    """
    return FeatureTransformer()


def apply_standard_scaling(
    data: pd.DataFrame,
    columns: List[str],
) -> Tuple[pd.DataFrame, FeatureTransformer]:
    """Apply standard scaling to specified columns.
    
    Args:
        data: Input data
        columns: Columns to scale
        
    Returns:
        Tuple of (scaled data, fitted transformer)
    """
    transformer = FeatureTransformer()
    transformer.add_transformation(
        "standard_scaler",
        TransformationType.STANDARD_SCALER,
        columns,
    )
    
    scaled_data = transformer.fit_transform(data)
    return scaled_data, transformer


def apply_log_transform(
    data: pd.DataFrame,
    columns: List[str],
    offset: float = 1.0,
) -> Tuple[pd.DataFrame, FeatureTransformer]:
    """Apply log transformation to specified columns.
    
    Args:
        data: Input data
        columns: Columns to transform
        offset: Offset to add before log transform
        
    Returns:
        Tuple of (transformed data, fitted transformer)
    """
    transformer = FeatureTransformer()
    transformer.add_transformation(
        "log_transform",
        TransformationType.LOG_TRANSFORM,
        columns,
        offset=offset,
    )
    
    transformed_data = transformer.fit_transform(data)
    return transformed_data, transformer
