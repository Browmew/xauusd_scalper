"""
Model training module for XAUUSD scalping system.
Implements training for LightGBM, XGBoost, and LSTM models with ensemble stacking.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from .architectures import create_lgbm_model, create_lstm_model


class ModelTrainer:
    """
    Comprehensive model trainer for the HFT scalping system.
    
    Supports training of LightGBM, XGBoost, and LSTM models with ensemble stacking.
    Implements walk-forward validation and proper time-series cross-validation.
    """
    
    def __init__(self, model_config: Dict[str, Any], output_dir: str = "models/"):
        """
        Initialize the model trainer.
        
        Args:
            model_config: Configuration dictionary for models
            output_dir: Directory to save trained models
        """
        self.model_config = model_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.trained_models = {}
        self.ensemble_model = None
        
        # Training metrics
        self.training_history = {}
        
        self.logger.info(f"ModelTrainer initialized with output_dir: {output_dir}")
    
    def train_ensemble(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train the complete ensemble model stack.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Dictionary containing training results and metrics
        """
        self.logger.info("Starting ensemble training process")
        
        results = {}
        
        # Prepare validation data if not provided
        if X_val is None or y_val is None:
            # Use time series split for validation
            tscv = TimeSeriesSplit(n_splits=5)
            train_idx, val_idx = list(tscv.split(X_train))[-1]  # Use last split
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            X_train = X_train[train_idx]
            y_train = y_train[train_idx]
        
        # Train primary model (LightGBM)
        self.logger.info("Training LightGBM model")
        lgbm_model, lgbm_metrics = self._train_lgbm(X_train, y_train, X_val, y_val)
        self.trained_models['lgbm'] = lgbm_model
        results['lgbm'] = lgbm_metrics
        
        # Train secondary models
        self.logger.info("Training XGBoost model")
        xgb_model, xgb_metrics = self._train_xgboost(X_train, y_train, X_val, y_val)
        self.trained_models['xgboost'] = xgb_model
        results['xgboost'] = xgb_metrics
        
        self.logger.info("Training LSTM model")
        lstm_model, lstm_metrics = self._train_lstm(X_train, y_train, X_val, y_val)
        self.trained_models['lstm'] = lstm_model
        results['lstm'] = lstm_metrics
        
        # Train ensemble meta-learner
        self.logger.info("Training ensemble meta-learner")
        ensemble_metrics = self._train_ensemble_meta_learner(X_train, y_train, X_val, y_val)
        results['ensemble'] = ensemble_metrics
        
        # Save all models
        self._save_models()
        
        self.logger.info("Ensemble training completed successfully")
        return results
    
    def _train_lgbm(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
        """
        Train LightGBM model with hyperparameters from config.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model and metrics
        """
        lgbm_config = self.model_config.get('primary', {}).get('lgbm', {})
        
        # Create and configure LightGBM model
        model = create_lgbm_model(lgbm_config)
        
        # Prepare datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        # Training parameters
        params = {
            'objective': lgbm_config.get('objective', 'binary'),
            'metric': lgbm_config.get('metric', 'binary_logloss'),
            'boosting_type': lgbm_config.get('boosting_type', 'gbdt'),
            'num_leaves': lgbm_config.get('num_leaves', 31),
            'learning_rate': lgbm_config.get('learning_rate', 0.05),
            'feature_fraction': lgbm_config.get('feature_fraction', 0.9),
            'bagging_fraction': lgbm_config.get('bagging_fraction', 0.8),
            'bagging_freq': lgbm_config.get('bagging_freq', 5),
            'verbose': -1,
            'random_state': lgbm_config.get('random_state', 42)
        }
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            num_boost_round=lgbm_config.get('n_estimators', 1000),
            callbacks=[
                lgb.early_stopping(lgbm_config.get('early_stopping_rounds', 100)),
                lgb.log_evaluation(period=0)  # Silent training
            ]
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_binary),
            'precision': precision_score(y_val, y_pred_binary, average='weighted'),
            'recall': recall_score(y_val, y_pred_binary, average='weighted'),
            'f1': f1_score(y_val, y_pred_binary, average='weighted'),
            'best_iteration': model.best_iteration
        }
        
        self.logger.info(f"LightGBM training completed. Best iteration: {model.best_iteration}")
        return model, metrics
    
    def _train_xgboost(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[xgb.XGBClassifier, Dict[str, float]]:
        """
        Train XGBoost model with hyperparameters from config.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Trained model and metrics
        """
        xgb_config = self.model_config.get('secondary', {}).get('xgboost', {})
        
        # Initialize XGBoost classifier
        model = xgb.XGBClassifier(
            n_estimators=xgb_config.get('n_estimators', 1000),
            max_depth=xgb_config.get('max_depth', 6),
            learning_rate=xgb_config.get('learning_rate', 0.05),
            subsample=xgb_config.get('subsample', 0.8),
            colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
            reg_alpha=xgb_config.get('reg_alpha', 0.0),
            reg_lambda=xgb_config.get('reg_lambda', 1.0),
            random_state=xgb_config.get('random_state', 42),
            n_jobs=xgb_config.get('n_jobs', -1),
            eval_metric='logloss',
            early_stopping_rounds=xgb_config.get('early_stopping_rounds', 100),
            verbose=False
        )
        
        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Calculate metrics
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted'),
            'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else model.n_estimators
        }
        
        self.logger.info(f"XGBoost training completed. Best iteration: {metrics['best_iteration']}")
        return model, metrics
    
    def _train_lstm(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Tuple[keras.Model, Dict[str, Any]]:
        """
        Train LSTM model with sequence reshaping and proper time-series handling.
        
        Args:
            X_train: Training features (2D: samples, features)
            y_train: Training labels
            X_val: Validation features (2D: samples, features)
            y_val: Validation labels
            
        Returns:
            Trained model and metrics
        """
        lstm_config = self.model_config.get('secondary', {}).get('lstm', {})
        
        # Get sequence parameters
        sequence_length = lstm_config.get('sequence_length', 60)
        batch_size = lstm_config.get('batch_size', 32)
        epochs = lstm_config.get('epochs', 100)
        patience = lstm_config.get('early_stopping_patience', 20)
        
        # Reshape data for LSTM (samples, timesteps, features)
        self.logger.info(f"Reshaping data for LSTM with sequence_length={sequence_length}")
        
        X_train_lstm, y_train_lstm = self._create_sequences(X_train, y_train, sequence_length)
        X_val_lstm, y_val_lstm = self._create_sequences(X_val, y_val, sequence_length)
        
        self.logger.info(f"LSTM input shape: {X_train_lstm.shape}")
        
        # Create LSTM model
        input_shape = (sequence_length, X_train_lstm.shape[2])
        model = create_lstm_model(input_shape, lstm_config)
        
        # Compile model
        optimizer = Adam(learning_rate=lstm_config.get('learning_rate', 0.001))
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience // 2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        self.logger.info("Starting LSTM training")
        history = model.fit(
            X_train_lstm, y_train_lstm,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val_lstm, y_val_lstm),
            callbacks=callbacks,
            verbose=1
        )
        
        # Calculate metrics
        y_pred_proba = model.predict(X_val_lstm)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_val_lstm, y_pred),
            'precision': precision_score(y_val_lstm, y_pred, average='weighted'),
            'recall': recall_score(y_val_lstm, y_pred, average='weighted'),
            'f1': f1_score(y_val_lstm, y_pred, average='weighted'),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'epochs_trained': len(history.history['loss'])
        }
        
        # Store training history
        self.training_history['lstm'] = history.history
        
        self.logger.info(f"LSTM training completed. Epochs trained: {metrics['epochs_trained']}")
        return model, metrics
    
    def _create_sequences(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        sequence_length: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create overlapping sequences for LSTM training.
        
        Args:
            X: Input features (2D)
            y: Target labels (1D)
            sequence_length: Length of each sequence
            
        Returns:
            Reshaped X and y for LSTM
        """
        if len(X) < sequence_length:
            raise ValueError(f"Not enough data points ({len(X)}) for sequence_length ({sequence_length})")
        
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(X)):
            X_sequences.append(X[i - sequence_length:i])
            y_sequences.append(y[i])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def _train_ensemble_meta_learner(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, float]:
        """
        Train the ensemble meta-learner using out-of-fold predictions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Ensemble metrics
        """
        ensemble_config = self.model_config.get('ensemble', {})
        
        # Generate out-of-fold predictions for meta-learner training
        self.logger.info("Generating out-of-fold predictions for ensemble")
        
        n_splits = ensemble_config.get('cv_folds', 5)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Storage for out-of-fold predictions
        oof_lgbm = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))
        oof_lstm = np.zeros(len(X_train))
        
        # Generate predictions for each fold
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            self.logger.info(f"Training ensemble fold {fold + 1}/{n_splits}")
            
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
            
            # Train temporary models for this fold
            lgbm_fold = create_lgbm_model(self.model_config.get('primary', {}).get('lgbm', {}))
            train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
            val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
            
            lgbm_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'verbose': -1,
                'random_state': 42
            }
            
            lgbm_fold = lgb.train(
                lgbm_params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
            )
            
            # XGBoost fold
            xgb_fold = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=20,
                verbose=False
            )
            xgb_fold.fit(X_fold_train, y_fold_train, eval_set=[(X_fold_val, y_fold_val)], verbose=False)
            
            # Generate predictions
            oof_lgbm[val_idx] = lgbm_fold.predict(X_fold_val, num_iteration=lgbm_fold.best_iteration)
            oof_xgb[val_idx] = xgb_fold.predict_proba(X_fold_val)[:, 1]
            
            # For LSTM, use simplified approach due to sequence requirements
            # In production, would implement proper sequence-aware CV
            oof_lstm[val_idx] = oof_lgbm[val_idx]  # Placeholder
        
        # Prepare meta-features
        meta_features_train = np.column_stack([oof_lgbm, oof_xgb, oof_lstm])
        
        # Generate validation predictions
        lgbm_val_pred = self.trained_models['lgbm'].predict(X_val, num_iteration=self.trained_models['lgbm'].best_iteration)
        xgb_val_pred = self.trained_models['xgboost'].predict_proba(X_val)[:, 1]
        
        # For LSTM, reshape validation data
        sequence_length = self.model_config.get('secondary', {}).get('lstm', {}).get('sequence_length', 60)
        if len(X_val) >= sequence_length:
            X_val_lstm, _ = self._create_sequences(X_val, y_val, sequence_length)
            lstm_val_pred = self.trained_models['lstm'].predict(X_val_lstm).flatten()
            
            # Pad LSTM predictions to match other models
            lstm_val_pred_full = np.zeros(len(X_val))
            lstm_val_pred_full[sequence_length:] = lstm_val_pred
            lstm_val_pred_full[:sequence_length] = lstm_val_pred[0]  # Fill with first prediction
        else:
            lstm_val_pred_full = lgbm_val_pred  # Fallback
        
        meta_features_val = np.column_stack([lgbm_val_pred, xgb_val_pred, lstm_val_pred_full])
        
        # Train meta-learner
        self.ensemble_model = LogisticRegression(
            random_state=ensemble_config.get('random_state', 42),
            max_iter=1000
        )
        
        self.ensemble_model.fit(meta_features_train, y_train)
        
        # Calculate ensemble metrics
        ensemble_pred = self.ensemble_model.predict(meta_features_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, ensemble_pred),
            'precision': precision_score(y_val, ensemble_pred, average='weighted'),
            'recall': recall_score(y_val, ensemble_pred, average='weighted'),
            'f1': f1_score(y_val, ensemble_pred, average='weighted')
        }
        
        self.logger.info(f"Ensemble meta-learner training completed. Accuracy: {metrics['accuracy']:.4f}")
        return metrics
    
    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        """
        Generate ensemble predictions using all trained models.
        
        Args:
            X: Input features
            
        Returns:
            Ensemble predictions
        """
        if self.ensemble_model is None:
            raise ValueError("Ensemble model not trained. Call train_ensemble first.")
        
        # Get predictions from all models
        lgbm_pred = self.trained_models['lgbm'].predict(X, num_iteration=self.trained_models['lgbm'].best_iteration)
        xgb_pred = self.trained_models['xgboost'].predict_proba(X)[:, 1]
        
        # Handle LSTM predictions with sequence requirements
        sequence_length = self.model_config.get('secondary', {}).get('lstm', {}).get('sequence_length', 60)
        if len(X) >= sequence_length:
            X_lstm, _ = self._create_sequences(X, np.zeros(len(X)), sequence_length)
            lstm_pred = self.trained_models['lstm'].predict(X_lstm).flatten()
            
            # Pad LSTM predictions
            lstm_pred_full = np.zeros(len(X))
            lstm_pred_full[sequence_length:] = lstm_pred
            lstm_pred_full[:sequence_length] = lstm_pred[0]
        else:
            lstm_pred_full = lgbm_pred  # Fallback
        
        # Combine predictions
        meta_features = np.column_stack([lgbm_pred, xgb_pred, lstm_pred_full])
        
        return self.ensemble_model.predict_proba(meta_features)[:, 1]
    
    def _save_models(self) -> None:
        """Save all trained models to disk."""
        self.logger.info("Saving trained models")
        
        # Save LightGBM
        if 'lgbm' in self.trained_models:
            lgbm_path = self.output_dir / "lgbm_model.txt"
            self.trained_models['lgbm'].save_model(str(lgbm_path))
            self.logger.info(f"LightGBM model saved to {lgbm_path}")
        
        # Save XGBoost
        if 'xgboost' in self.trained_models:
            xgb_path = self.output_dir / "xgboost_model.json"
            self.trained_models['xgboost'].save_model(str(xgb_path))
            self.logger.info(f"XGBoost model saved to {xgb_path}")
        
        # Save LSTM
        if 'lstm' in self.trained_models:
            lstm_path = self.output_dir / "lstm_model"
            self.trained_models['lstm'].save(str(lstm_path))
            self.logger.info(f"LSTM model saved to {lstm_path}")
        
        # Save ensemble meta-learner
        if self.ensemble_model is not None:
            ensemble_path = self.output_dir / "ensemble_model.pkl"
            joblib.dump(self.ensemble_model, ensemble_path)
            self.logger.info(f"Ensemble model saved to {ensemble_path}")
        
        # Save training history
        if self.training_history:
            history_path = self.output_dir / "training_history.pkl"
            joblib.dump(self.training_history, history_path)
            self.logger.info(f"Training history saved to {history_path}")
    
    def load_models(self) -> None:
        """Load previously trained models from disk."""
        self.logger.info("Loading trained models")
        
        try:
            # Load LightGBM
            lgbm_path = self.output_dir / "lgbm_model.txt"
            if lgbm_path.exists():
                self.trained_models['lgbm'] = lgb.Booster(model_file=str(lgbm_path))
                self.logger.info("LightGBM model loaded")
            
            # Load XGBoost
            xgb_path = self.output_dir / "xgboost_model.json"
            if xgb_path.exists():
                self.trained_models['xgboost'] = xgb.XGBClassifier()
                self.trained_models['xgboost'].load_model(str(xgb_path))
                self.logger.info("XGBoost model loaded")
            
            # Load LSTM
            lstm_path = self.output_dir / "lstm_model"
            if Path(lstm_path).exists():
                self.trained_models['lstm'] = keras.models.load_model(str(lstm_path))
                self.logger.info("LSTM model loaded")
            
            # Load ensemble meta-learner
            ensemble_path = self.output_dir / "ensemble_model.pkl"
            if ensemble_path.exists():
                self.ensemble_model = joblib.load(ensemble_path)
                self.logger.info("Ensemble model loaded")
            
            # Load training history
            history_path = self.output_dir / "training_history.pkl"
            if history_path.exists():
                self.training_history = joblib.load(history_path)
                self.logger.info("Training history loaded")
                
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
            raise