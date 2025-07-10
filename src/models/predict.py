"""
Prediction module for the XAUUSD scalping system.

This module provides the ModelPredictor class for loading trained models
and running efficient inference on new data.
"""

import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass

from ..utils.helpers import get_config_value, get_project_root
from ..utils.logging import get_logger


@dataclass
class PredictionResult:
    """
    Container for prediction results.
    
    Attributes:
        prediction: The model's prediction value
        confidence: Model confidence score (if available)
        features_used: Number of features used in prediction
        timestamp: Timestamp of the prediction
    """
    prediction: float
    confidence: Optional[float] = None
    features_used: int = 0
    timestamp: Optional[pd.Timestamp] = None


class ModelPredictor:
    """
    High-performance model prediction interface for XAUUSD scalping.
    
    This class loads trained model artifacts and their metadata to provide
    fast, consistent predictions on feature-engineered data. It ensures
    feature alignment and handles missing data gracefully.
    """
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the ModelPredictor.
        
        Args:
            model_path: Path to the model file. If None, loads the latest model.
        """
        self.logger = get_logger(__name__)
        self.model = None
        self.metadata = None
        self.feature_names: List[str] = []
        self.model_type: str = ""
        self.model_path: str = ""
        
        if model_path:
            self.load_model(model_path)
        else:
            self._load_latest_model()
    
    def _load_latest_model(self) -> None:
        """Load the most recently trained model."""
        models_dir = get_project_root() / "models" / "trained"
        
        if not models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find the latest model file
        model_files = list(models_dir.glob("*.pkl"))
        if not model_files:
            raise FileNotFoundError(f"No trained models found in {models_dir}")
        
        # Sort by modification time and get the latest
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        self.load_model(str(latest_model))
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model and its metadata.
        """
        model_path_obj = Path(model_path)
        
        if not model_path_obj.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load the model
        try:
            self.model = joblib.load(model_path)
            self.model_path = str(model_path_obj)
            self.logger.info(f"Loaded model from {model_path}")
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # ------------------------------------------------------------------
        # Load feature-order metadata if present
        # ------------------------------------------------------------------
        feature_list_path = Path(model_path).with_name(
            Path(model_path).stem + "_features.json"
        )
        self.expected_features: list[str] = []
        if feature_list_path.exists():
            self.expected_features = json.loads(feature_list_path.read_text())

        
        # Load metadata - rest stays the same
        metadata_path = model_path_obj.parent / (model_path_obj.stem + '_metadata.json')

        if not metadata_path.exists():
            self.logger.warning(f"Model metadata not found: {metadata_path}")
            self.metadata = {
                'feature_names': [],
                'model_type': 'unknown'
            }
            self.feature_names = []
            self.model_type = 'unknown'
            return

        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            
            required_fields = ['feature_names', 'model_type']
            for field in required_fields:
                if field not in self.metadata:
                    raise ValueError(f"Missing required metadata field: {field}")
            
            self.feature_names = self.metadata['feature_names']
            self.model_type = self.metadata['model_type']
            
            self.logger.info(f"Loaded metadata: {len(self.feature_names)} features, type: {self.model_type}")
            
        except Exception as e:
            error_msg = f"Failed to load or validate metadata: {e}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare and align features for prediction.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            DataFrame with features aligned to training expectations
            
        Raises:
            ValueError: If required features are missing
        """
        if not self.feature_names:
            # Use all available columns if no feature names specified
            self.feature_names = list(data.columns)
        
        # Check for missing features
        missing_features = set(self.feature_names) - set(data.columns)
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select and order features as expected by the model
        feature_data = data[self.feature_names].copy()
        
        # Handle any NaN values by forward filling then backward filling
        feature_data = feature_data.fillna(method='ffill').fillna(method='bfill')
        
        # If still NaN, fill with 0 (should be rare with proper feature engineering)
        feature_data = feature_data.fillna(0)
        
        return feature_data
    
    def predict(self, data: pd.DataFrame) -> Union[float, np.ndarray]:
        """
        Make predictions on the provided data.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            Prediction(s) - single value for one row, array for multiple rows
            
        Raises:
            ValueError: If model not loaded or data is invalid
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if data.empty:
            raise ValueError("Input data is empty")
        
        # Align column order if we have metadata
        if self.expected_features:
            missing = set(self.expected_features) - set(X.columns)
            if missing:
                raise ValueError(f"Missing features for prediction: {missing}")
            X = X[self.expected_features]  # re-index & order

        
        # Prepare features
        feature_data = self._prepare_features(data)
        
        # Make prediction
        try:
            predictions = self.model.predict(feature_data)
            
            # Return single value for single row, otherwise return array
            if len(predictions) == 1:
                return float(predictions[0])
            return predictions
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise ValueError(f"Prediction failed: {e}")
    
    def predict_with_metadata(self, data: pd.DataFrame) -> Union[PredictionResult, List[PredictionResult]]:
        """
        Make predictions with additional metadata.
        
        Args:
            data: DataFrame with feature columns
            
        Returns:
            PredictionResult or list of PredictionResults
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        feature_data = self._prepare_features(data)
        predictions = self.model.predict(feature_data)
        
        # Get confidence scores if available (for models that support it)
        confidence_scores = None
        if hasattr(self.model, 'predict_proba'):
            try:
                proba = self.model.predict_proba(feature_data)
                # For binary classification, use max probability as confidence
                confidence_scores = np.max(proba, axis=1)
            except:
                confidence_scores = None
        
        # Create results
        results = []
        for i, pred in enumerate(predictions):
            timestamp = data.index[i] if hasattr(data.index, '__getitem__') else None
            confidence = confidence_scores[i] if confidence_scores is not None else None
            
            result = PredictionResult(
                prediction=float(pred),
                confidence=confidence,
                features_used=len(self.feature_names),
                timestamp=timestamp
            )
            results.append(result)
        
        # Return single result for single row
        if len(results) == 1:
            return results[0]
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "model_path": self.model_path,
            "model_type": self.model_type,
            "feature_count": len(self.feature_names),
            "feature_names": self.feature_names,
        }
        
        # Add metadata if available
        if self.metadata:
            info.update({
                "training_score": self.metadata.get("training_score"),
                "validation_score": self.metadata.get("validation_score"),
                "training_samples": self.metadata.get("training_samples"),
                "trained_at": self.metadata.get("trained_at"),
            })
        
        return info