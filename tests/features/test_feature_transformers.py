"""Tests for feature transformers module."""

from __future__ import annotations

import pytest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from astroml.features.feature_transformers import (
    FeatureTransformer,
    TransformationType,
    TransformationConfig,
    LogTransformer,
    Bucketizer,
    FeatureEngineering,
    create_feature_transformer,
    apply_standard_scaling,
    apply_log_transform,
)


class TestTransformationConfig:
    """Test TransformationConfig class."""
    
    def test_transformation_config_creation(self):
        """Test creating transformation configuration."""
        config = TransformationConfig(
            transformation_type=TransformationType.STANDARD_SCALER,
            parameters={"with_mean": True},
            input_columns=["feature1", "feature2"],
            output_columns=["scaled_feature1", "scaled_feature2"],
        )
        
        assert config.transformation_type == TransformationType.STANDARD_SCALER
        assert config.parameters["with_mean"] is True
        assert config.input_columns == ["feature1", "feature2"]
        assert config.output_columns == ["scaled_feature1", "scaled_feature2"]


class TestLogTransformer:
    """Test LogTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "positive_values": [1.0, 10.0, 100.0, 1000.0],
            "zero_values": [0.0, 0.0, 1.0, 10.0],
            "negative_values": [-1.0, -10.0, 1.0, 10.0],
        })
    
    def test_log_transform_positive_values(self, sample_data):
        """Test log transformation on positive values."""
        transformer = LogTransformer(offset=1.0)
        result = transformer.fit_transform(sample_data[["positive_values"]])
        
        expected = np.log(sample_data["positive_values"] + 1.0)
        np.testing.assert_array_almost_equal(result["positive_values"], expected)
    
    def test_log_transform_with_zero(self, sample_data):
        """Test log transformation with zero values."""
        transformer = LogTransformer(offset=1.0)
        result = transformer.fit_transform(sample_data[["zero_values"]])
        
        # Should handle zeros correctly with offset
        assert not result.isnull().any().any()
    
    def test_log_transform_negative_error(self, sample_data):
        """Test log transformer with negative values raises error."""
        transformer = LogTransformer(handle_negative="error")
        
        with pytest.raises(ValueError, match="Negative values found"):
            transformer.fit_transform(sample_data[["negative_values"]])
    
    def test_log_transform_negative_abs(self, sample_data):
        """Test log transformer with negative values using absolute value."""
        transformer = LogTransformer(handle_negative="abs")
        result = transformer.fit_transform(sample_data[["negative_values"]])
        
        # Should handle negative values by taking absolute
        assert not result.isnull().any().any()
    
    def test_log_transform_negative_clip(self, sample_data):
        """Test log transformer with negative values using clipping."""
        transformer = LogTransformer(handle_negative="clip")
        result = transformer.fit_transform(sample_data[["negative_values"]])
        
        # Should handle negative values by clipping to zero
        assert not result.isnull().any().any()


class TestBucketizer:
    """Test Bucketizer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        return pd.DataFrame({
            "values": np.random.normal(0, 1, 100),
            "categories": np.random.choice(["A", "B", "C"], 100),
        })
    
    def test_bucketizer_uniform(self, sample_data):
        """Test bucketizer with uniform strategy."""
        bucketizer = Bucketizer(n_bins=5, strategy="uniform")
        result = bucketizer.fit_transform(sample_data[["values"]])
        
        assert len(result["values"].unique()) <= 5
        assert not result["values"].isnull().any()
    
    def test_bucketizer_quantile(self, sample_data):
        """Test bucketizer with quantile strategy."""
        bucketizer = Bucketizer(n_bins=4, strategy="quantile")
        result = bucketizer.fit_transform(sample_data[["values"]])
        
        assert len(result["values"].unique()) <= 4
        assert not result["values"].isnull().any()
    
    def test_bucketizer_with_labels(self, sample_data):
        """Test bucketizer with custom labels."""
        labels = ["very_low", "low", "medium", "high"]
        bucketizer = Bucketizer(n_bins=4, strategy="quantile", labels=labels)
        result = bucketizer.fit_transform(sample_data[["values"]])
        
        # Should use custom labels
        unique_values = result["values"].unique()
        for label in labels[:len(unique_values)]:
            assert label in unique_values


class TestFeatureTransformer:
    """Test FeatureTransformer class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame({
            "numeric1": np.random.normal(0, 1, 100),
            "numeric2": np.random.exponential(1, 100),
            "categorical": np.random.choice(["A", "B", "C"], 100),
            "binary": np.random.choice([0, 1], 100),
        })
    
    def test_add_transformation(self, sample_data):
        """Test adding transformations."""
        transformer = FeatureTransformer()
        
        transformer.add_transformation(
            "standard_scaler",
            TransformationType.STANDARD_SCALER,
            ["numeric1", "numeric2"],
        )
        
        assert "standard_scaler" in transformer.list_transformations()
        config = transformer.get_config("standard_scaler")
        assert config.transformation_type == TransformationType.STANDARD_SCALER
        assert config.input_columns == ["numeric1", "numeric2"]
    
    def test_fit_transform(self, sample_data):
        """Test fit and transform operations."""
        transformer = FeatureTransformer()
        
        # Add standard scaling
        transformer.add_transformation(
            "standard_scaler",
            TransformationType.STANDARD_SCALER,
            ["numeric1", "numeric2"],
        )
        
        # Fit and transform
        result = transformer.fit_transform(sample_data)
        
        # Check that numeric columns are scaled
        assert result["numeric1"].mean() == pytest.approx(0, abs=1e-10)
        assert result["numeric2"].mean() == pytest.approx(0, abs=1e-10)
        assert result["numeric1"].std() == pytest.approx(1, abs=1e-10)
        assert result["numeric2"].std() == pytest.approx(1, abs=1e-10)
        
        # Check that other columns are unchanged
        assert list(result["categorical"].unique()) == ["A", "B", "C"]
        assert set(result["binary"].unique()) == {0, 1}
    
    def test_multiple_transformations(self, sample_data):
        """Test applying multiple transformations."""
        transformer = FeatureTransformer()
        
        # Add standard scaling
        transformer.add_transformation(
            "standard_scaler",
            TransformationType.STANDARD_SCALER,
            ["numeric1"],
        )
        
        # Add log transformation
        transformer.add_transformation(
            "log_transform",
            TransformationType.LOG_TRANSFORM,
            ["numeric2"],
            offset=1.0,
        )
        
        # Fit and transform
        result = transformer.fit_transform(sample_data)
        
        # Check transformations were applied
        assert result["numeric1"].mean() == pytest.approx(0, abs=1e-10)
        assert result["numeric2"].min() >= 0  # Log transform should make values non-negative
    
    def test_remove_transformation(self, sample_data):
        """Test removing transformations."""
        transformer = FeatureTransformer()
        
        # Add transformation
        transformer.add_transformation(
            "standard_scaler",
            TransformationType.STANDARD_SCALER,
            ["numeric1"],
        )
        
        assert "standard_scaler" in transformer.list_transformations()
        
        # Remove transformation
        transformer.remove_transformation("standard_scaler")
        assert "standard_scaler" not in transformer.list_transformations()
    
    def test_save_load_transformer(self, sample_data):
        """Test saving and loading transformer."""
        import tempfile
        import os
        
        transformer = FeatureTransformer()
        transformer.add_transformation(
            "standard_scaler",
            TransformationType.STANDARD_SCALER,
            ["numeric1"],
        )
        
        # Fit transformer
        transformer.fit(sample_data)
        
        # Save transformer
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            transformer.save(temp_path)
            
            # Load transformer
            loaded_transformer = FeatureTransformer.load(temp_path)
            
            # Check that loaded transformer works
            result = loaded_transformer.transform(sample_data)
            assert result["numeric1"].mean() == pytest.approx(0, abs=1e-10)
            
        finally:
            os.unlink(temp_path)


class TestFeatureEngineering:
    """Test FeatureEngineering class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.exponential(1, 100),
            "feature3": np.random.uniform(0, 10, 100),
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="D"),
        })
    
    def test_create_interaction_features(self, sample_data):
        """Test creating interaction features."""
        result = FeatureEngineering.create_interaction_features(
            sample_data,
            ["feature1", "feature2"],
            interaction_type="multiplication",
        )
        
        # Check interaction column was created
        assert "feature1_x_feature2" in result.columns
        assert "feature2_x_feature1" not in result.columns  # Should not create duplicate
        
        # Check interaction values
        expected = sample_data["feature1"] * sample_data["feature2"]
        pd.testing.assert_series_equal(result["feature1_x_feature2"], expected)
    
    def test_create_interaction_features_addition(self, sample_data):
        """Test creating addition interaction features."""
        result = FeatureEngineering.create_interaction_features(
            sample_data,
            ["feature1", "feature2"],
            interaction_type="addition",
        )
        
        assert "feature1_plus_feature2" in result.columns
        expected = sample_data["feature1"] + sample_data["feature2"]
        pd.testing.assert_series_equal(result["feature1_plus_feature2"], expected)
    
    def test_create_interaction_features_subtraction(self, sample_data):
        """Test creating subtraction interaction features."""
        result = FeatureEngineering.create_interaction_features(
            sample_data,
            ["feature1", "feature2"],
            interaction_type="subtraction",
        )
        
        # Should create both subtraction directions
        assert "feature1_minus_feature2" in result.columns
        assert "feature2_minus_feature1" in result.columns
        
        # Check values
        expected1 = sample_data["feature1"] - sample_data["feature2"]
        expected2 = sample_data["feature2"] - sample_data["feature1"]
        pd.testing.assert_series_equal(result["feature1_minus_feature2"], expected1)
        pd.testing.assert_series_equal(result["feature2_minus_feature1"], expected2)
    
    def test_create_polynomial_features(self, sample_data):
        """Test creating polynomial features."""
        result = FeatureEngineering.create_polynomial_features(
            sample_data,
            ["feature1"],
            degree=2,
        )
        
        # Should have polynomial features
        assert "feature1^2" in result.columns
        
        # Check polynomial values
        expected = sample_data["feature1"] ** 2
        pd.testing.assert_series_equal(result["feature1^2"], expected)
    
    def test_create_rolling_features(self, sample_data):
        """Test creating rolling features."""
        # Set timestamp as index for rolling features
        data_with_index = sample_data.set_index("timestamp")
        
        result = FeatureEngineering.create_rolling_features(
            data_with_index,
            ["feature1"],
            window_sizes=[5],
            functions=["mean", "std"],
        )
        
        # Check rolling features were created
        assert "feature1_rolling_5_mean" in result.columns
        assert "feature1_rolling_5_std" in result.columns
        
        # Rolling features should have NaN values at the beginning
        assert result["feature1_rolling_5_mean"].isna().sum() > 0
    
    def test_create_lag_features(self, sample_data):
        """Test creating lag features."""
        # Set timestamp as index for lag features
        data_with_index = sample_data.set_index("timestamp")
        
        result = FeatureEngineering.create_lag_features(
            data_with_index,
            ["feature1"],
            lags=[1, 2],
        )
        
        # Check lag features were created
        assert "feature1_lag_1" in result.columns
        assert "feature1_lag_2" in result.columns
        
        # Lag features should have NaN values at the beginning
        assert result["feature1_lag_1"].isna().sum() > 0
        assert result["feature1_lag_2"].isna().sum() > 0
    
    def test_create_time_features(self, sample_data):
        """Test creating time features."""
        result = FeatureEngineering.create_time_features(
            sample_data,
            "timestamp",
        )
        
        # Check time features were created
        time_features = [
            "hour", "day_of_week", "day_of_month", "month", 
            "quarter", "year", "hour_sin", "hour_cos",
            "day_sin", "day_cos", "month_sin", "month_cos", "is_weekend"
        ]
        
        for feature in time_features:
            assert feature in result.columns
        
        # Check value ranges
        assert result["hour"].between(0, 23).all()
        assert result["day_of_week"].between(0, 6).all()
        assert result["month"].between(1, 12).all()
        assert result["is_weekend"].between(0, 1).all()
    
    def test_detect_outliers_iqr(self, sample_data):
        """Test outlier detection using IQR method."""
        result = FeatureEngineering.detect_outliers(
            sample_data,
            ["feature1"],
            method="iqr",
            threshold=1.5,
        )
        
        # Check outlier column was created
        assert "feature1_outlier" in result.columns
        
        # Check outlier values are 0 or 1
        outliers = result["feature1_outlier"]
        assert set(outliers.unique()).issubset({0, 1})
    
    def test_detect_outliers_zscore(self, sample_data):
        """Test outlier detection using Z-score method."""
        result = FeatureEngineering.detect_outliers(
            sample_data,
            ["feature1"],
            method="zscore",
            threshold=2.0,
        )
        
        # Check outlier column was created
        assert "feature1_outlier" in result.columns
        
        # Check outlier values are 0 or 1
        outliers = result["feature1_outlier"]
        assert set(outliers.unique()).issubset({0, 1})
    
    def test_detect_outliers_isolation_forest(self, sample_data):
        """Test outlier detection using Isolation Forest."""
        result = FeatureEngineering.detect_outliers(
            sample_data,
            ["feature1"],
            method="isolation_forest",
        )
        
        # Check outlier column was created
        assert "feature1_outlier" in result.columns
        
        # Check outlier values are 0 or 1
        outliers = result["feature1_outlier"]
        assert set(outliers.unique()).issubset({0, 1})


class TestConvenienceFunctions:
    """Test convenience functions."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data."""
        np.random.seed(42)
        return pd.DataFrame({
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.exponential(1, 100),
        })
    
    def test_create_feature_transformer(self):
        """Test create_feature_transformer function."""
        transformer = create_feature_transformer()
        assert isinstance(transformer, FeatureTransformer)
        assert len(transformer.list_transformations()) == 0
    
    def test_apply_standard_scaling(self, sample_data):
        """Test apply_standard_scaling function."""
        scaled_data, transformer = apply_standard_scaling(
            sample_data,
            ["feature1", "feature2"],
        )
        
        # Check that data was scaled
        assert scaled_data["feature1"].mean() == pytest.approx(0, abs=1e-10)
        assert scaled_data["feature2"].mean() == pytest.approx(0, abs=1e-10)
        
        # Check that transformer was fitted
        assert transformer._fitted is True
    
    def test_apply_log_transform(self, sample_data):
        """Test apply_log_transform function."""
        # Use only positive data for log transform
        positive_data = sample_data.copy()
        positive_data["feature1"] = np.abs(positive_data["feature1"]) + 1.0
        positive_data["feature2"] = np.abs(positive_data["feature2"]) + 1.0
        
        transformed_data, transformer = apply_log_transform(
            positive_data,
            ["feature1", "feature2"],
            offset=1.0,
        )
        
        # Check that data was log transformed
        assert transformed_data["feature1"].min() >= 0
        assert transformed_data["feature2"].min() >= 0
        
        # Check that transformer was fitted
        assert transformer._fitted is True


if __name__ == "__main__":
    pytest.main([__file__])
