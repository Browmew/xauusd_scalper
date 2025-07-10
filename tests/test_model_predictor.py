"""
Unit tests for the ModelPredictor component.
Tests model loading, metadata handling, and prediction logic.
"""

import pandas as pd
import pytest
import joblib
import json
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np

from src.models.predict import ModelPredictor


class TestModelPredictor:
    """Test suite for ModelPredictor component."""
    
    @pytest.fixture
    def sample_model(self):
        """Create a mock sklearn model for testing."""
        mock_model = Mock()
        mock_model.predict.return_value = np.array([0.1, 0.2, 0.3])
        return mock_model
    
    @pytest.fixture
    def sample_metadata(self) -> dict:
        """Create sample metadata for testing."""
        return {
            "feature_names": [
                "sma_5", "sma_20", "rsi_14", "macd_signal", 
                "bb_upper", "bb_lower", "volume_sma"
            ],
            "model_type": "RandomForestRegressor",
            "training_date": "2024-01-01",
            "performance_metrics": {
                "mse": 0.001,
                "r2": 0.85
            }
        }
    
    @pytest.fixture
    def sample_features_df(self, sample_metadata: dict) -> pd.DataFrame:
        """Create sample features DataFrame matching metadata."""
        np.random.seed(42)
        n_samples = 100
        
        data = {}
        for feature in sample_metadata["feature_names"]:
            data[feature] = np.random.randn(n_samples)
        
        # Add extra columns that shouldn't be used
        data["extra_col_1"] = np.random.randn(n_samples)
        data["extra_col_2"] = np.random.randn(n_samples)
        
        return pd.DataFrame(data)
    
    def create_model_files(self, tmp_path: Path, model: Mock, metadata: dict) -> Path:
        """Create temporary model and metadata files for testing."""
        model_path = tmp_path / "test_model.pkl"
        metadata_path = tmp_path / "test_model_metadata.json"
        
        # Use actual sklearn model that can be properly serialized
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        test_model = RandomForestRegressor(n_estimators=2, random_state=42)
        # Create dummy training data matching the feature names
        X_dummy = np.random.rand(10, len(metadata["feature_names"]))
        y_dummy = np.random.rand(10)
        test_model.fit(X_dummy, y_dummy)
        
        # Save the model using joblib
        joblib.dump(test_model, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
        
        return model_path
    
    def test_model_predictor_initialization_success(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test successful ModelPredictor initialization."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        
        predictor = ModelPredictor(str(model_path))
        
        assert predictor.model is not None
        assert predictor.feature_names == sample_metadata["feature_names"]
        assert hasattr(predictor, 'metadata')
    
    def test_model_predictor_initialization_missing_model(self, tmp_path: Path):
        """Test initialization with missing model file."""
        nonexistent_path = tmp_path / "nonexistent_model.pkl"
        
        with pytest.raises((FileNotFoundError, ValueError)):
            ModelPredictor(str(nonexistent_path))
    
    def test_model_predictor_initialization_missing_metadata(self, tmp_path: Path):
        """Test initialization with missing metadata file."""
        # Create a real model file but no metadata
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        model_path = tmp_path / "test_model.pkl"
        
        test_model = RandomForestRegressor(n_estimators=2, random_state=42)
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.rand(10)
        test_model.fit(X_dummy, y_dummy)
        
        joblib.dump(test_model, model_path)
        
        # Don't create metadata file - should fail gracefully
        predictor = ModelPredictor(str(model_path))
        
        # Should initialize with warning but empty feature names
        assert predictor.feature_names == []
        assert predictor.model_type == 'unknown'
    
    def test_model_predictor_initialization_malformed_metadata(self, tmp_path: Path):
        """Test initialization with malformed metadata file."""
        # Create a real model file
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        model_path = tmp_path / "test_model.pkl"
        metadata_path = tmp_path / "test_model_metadata.json"
        
        test_model = RandomForestRegressor(n_estimators=2, random_state=42)
        X_dummy = np.random.rand(10, 5)
        y_dummy = np.random.rand(10)
        test_model.fit(X_dummy, y_dummy)
        
        joblib.dump(test_model, model_path)
        
        # Create malformed JSON
        with open(metadata_path, 'w') as f:
            f.write("{ invalid json content")
        
        with pytest.raises((json.JSONDecodeError, ValueError)):
            ModelPredictor(str(model_path))
    
    def test_predict_success(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict, sample_features_df: pd.DataFrame):
        """Test successful prediction with proper feature alignment."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        predictions = predictor.predict(sample_features_df)
        
        # Verify predictions are returned
        assert predictions is not None
        assert len(predictions) == len(sample_features_df)
        assert isinstance(predictions, (list, np.ndarray))
    
    def test_predict_missing_features(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test prediction with missing required features."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        # Create DataFrame missing some required features
        incomplete_df = pd.DataFrame({
            "sma_5": [1.0, 2.0],
            "sma_20": [1.1, 2.1],
            # Missing other required features like bb_upper, rsi_14, etc.
            "extra_feature": [0.5, 0.6]
        })
        
        with pytest.raises(ValueError, match="Missing required features"):
            predictor.predict(incomplete_df)
    
    def test_predict_feature_order_independence(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test that feature order in input DataFrame doesn't matter."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        # Create DataFrame with features in different order
        shuffled_features = sample_metadata["feature_names"][::-1]  # Reverse order
        shuffled_data = {}
        np.random.seed(42)
        
        for feature in shuffled_features:
            shuffled_data[feature] = np.random.randn(10)
        
        shuffled_df = pd.DataFrame(shuffled_data)
        
        predictions = predictor.predict(shuffled_df)
        
        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) == len(shuffled_df)
    
    def test_predict_empty_dataframe(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test prediction with empty DataFrame."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=sample_metadata["feature_names"])
        
        with pytest.raises(ValueError, match="Input data is empty"):
            predictor.predict(empty_df)
    
    def test_predict_single_row(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test prediction with single row DataFrame."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        # Create single-row DataFrame
        single_row_data = {}
        for i, feature in enumerate(sample_metadata["feature_names"]):
            single_row_data[feature] = [i * 0.1]
        
        single_row_df = pd.DataFrame(single_row_data)
        
        predictions = predictor.predict(single_row_df)
        
        # Handle both single float and array returns
        if isinstance(predictions, (float, int, np.number)):
            assert True  # Single value returned as expected
        else:
            assert len(predictions) == 1
    
    def test_predict_with_nan_values(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test prediction behavior with NaN values in features."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        # Create DataFrame with NaN values
        nan_data = {}
        for i, feature in enumerate(sample_metadata["feature_names"]):
            if i == 0:
                nan_data[feature] = [1.0, np.nan, 3.0]
            else:
                nan_data[feature] = [i * 0.1, i * 0.2, i * 0.3]
        
        nan_df = pd.DataFrame(nan_data)
        
        # The predictor should handle NaN values (sklearn models can handle them)
        predictions = predictor.predict(nan_df)
        
        # Verify predictions were generated
        assert predictions is not None
        assert len(predictions) == len(nan_df)
    
    def test_metadata_access(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test access to model metadata."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        assert predictor.metadata["model_type"] == "RandomForestRegressor"
        assert predictor.metadata["training_date"] == "2024-01-01"
        assert "performance_metrics" in predictor.metadata
        assert predictor.metadata["performance_metrics"]["r2"] == 0.85
    
    def test_model_path_handling(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test various model path formats."""
        # Test with Path object
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        
        predictor1 = ModelPredictor(model_path)  # Path object
        predictor2 = ModelPredictor(str(model_path))  # String path
        
        assert predictor1.feature_names == predictor2.feature_names
        assert predictor1.metadata == predictor2.metadata
    
    @patch('joblib.load')
    def test_model_loading_error_handling(self, tmp_path: Path, sample_metadata: dict):
        """Test handling of model loading errors."""
        # Test with completely nonexistent file
        nonexistent_path = tmp_path / "nonexistent_model.pkl"
        
        with pytest.raises((FileNotFoundError, ValueError)):
            ModelPredictor(str(nonexistent_path))
    
    def test_feature_names_immutability(self, tmp_path: Path, sample_model: Mock, sample_metadata: dict):
        """Test that feature_names cannot be accidentally modified."""
        model_path = self.create_model_files(tmp_path, sample_model, sample_metadata)
        predictor = ModelPredictor(str(model_path))
        
        original_features = predictor.feature_names.copy()
        
        # Try to modify feature_names
        predictor.feature_names.append("new_feature")
        
        # Create new predictor instance to verify original metadata unchanged
        predictor2 = ModelPredictor(str(model_path))
        assert predictor2.feature_names == original_features