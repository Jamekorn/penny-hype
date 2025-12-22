import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import json
from typing import Dict, List, Tuple
from loguru import logger

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, f1_score
)

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

import mlflow
import mlflow.sklearn

class StockPredictionModel:
    def __init__(self, model_type: str = "xgboost"):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        
        if model_type == "xgboost":
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42
            )
        elif model_type == "lightgbm":
            self.model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42
            )
    
    def prepare_features(self, features_df: pd.DataFrame, 
                        target_col: str = "target") -> Tuple:
        """Prepare features for training"""
        if target_col not in features_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in dataframe")
        
        # Separate features and target
        X = features_df.drop(columns=[target_col])
        y = features_df[target_col]
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Handle missing values
        X = X.fillna(0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model"""
        logger.info(f"Training {self.model_type} model...")
        
        # If validation data not provided, split training data
        if X_val is None or y_val is None:
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
        
        # Train model
        if self.model_type in ["xgboost", "lightgbm"]:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=10,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_pred = self.model.predict(X_val)
        val_proba = self.model.predict_proba(X_val)[:, 1]
        
        metrics = {
            "accuracy": (val_pred == y_val).mean(),
            "precision": precision_score(y_val, val_pred, average='weighted'),
            "f1": f1_score(y_val, val_pred, average='weighted'),
            "roc_auc": roc_auc_score(y_val, val_proba) if len(set(y_val)) > 1 else 0.5
        }
        
        logger.info(f"Validation metrics: {metrics}")
        
        return metrics
    
    def predict(self, features: np.ndarray) -> Tuple:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        return predictions, probabilities
    
    def save(self, path: str):
        """Save model and scaler"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model and scaler"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        
        logger.info(f"Model loaded from {path}")

class ModelTrainer:
    def __init__(self, mlflow_tracking_uri: str = None):
        self.mlflow_tracking_uri = mlflow_tracking_uri or "mlruns"
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        self.short_term_model = None
        self.long_term_model = None
    
    def create_labels(self, price_data: pd.DataFrame, 
                     horizon_days: int = 1) -> pd.Series:
        """Create labels based on price movement"""
        if 'close' not in price_data.columns:
            raise ValueError("Price data must contain 'close' column")
        
        # Calculate future returns
        price_data = price_data.sort_index()
        price_data['future_close'] = price_data['close'].shift(-horizon_days)
        
        # Calculate return
        price_data['return'] = (price_data['future_close'] - price_data['close']) / price_data['close']
        
        # Create labels: 1 for positive return, 0 for negative
        labels = (price_data['return'] > 0).astype(int)
        
        return labels
    
    def prepare_training_data(self, features: Dict, labels: pd.Series) -> pd.DataFrame:
        """Prepare combined training data"""
        # Convert features dict to dataframe
        features_df = pd.DataFrame([features]).T
        features_df = features_df.T
        
        # Add labels
        features_df['target'] = labels
        
        return features_df
    
    def train_short_term_model(self, features_df: pd.DataFrame, 
                             model_type: str = "xgboost"):
        """Train short-term prediction model"""
        logger.info("Training short-term model...")
        
        # Use time series split for financial data
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features_df)):
            logger.info(f"Training fold {fold + 1}/5")
            
            # Split data
            train_data = features_df.iloc[train_idx]
            val_data = features_df.iloc[val_idx]
            
            # Prepare features
            X_train = train_data.drop(columns=['target'])
            y_train = train_data['target']
            X_val = val_data.drop(columns=['target'])
            y_val = val_data['target']
            
            # Handle missing values
            X_train = X_train.fillna(0)
            X_val = X_val.fillna(0)
            
            # Train model
            model = StockPredictionModel(model_type)
            metrics = model.train(X_train.values, y_train.values, 
                                X_val.values, y_val.values)
            
            all_metrics.append(metrics)
            
            # Save the last fold model
            if fold == 4:  # Last fold
                self.short_term_model = model
        
        # Calculate average metrics
        avg_metrics = {}
        for metric in all_metrics[0].keys():
            values = [m[metric] for m in all_metrics]
            avg_metrics[f"avg_{metric}"] = np.mean(values)
            avg_metrics[f"std_{metric}"] = np.std(values)
        
        logger.info(f"Short-term model training completed. Average metrics: {avg_metrics}")
        
        return avg_metrics
    
    def train_long_term_model(self, features_df: pd.DataFrame,
                            model_type: str = "lightgbm"):
        """Train long-term prediction model"""
        logger.info("Training long-term model...")
        
        # Similar to short-term but with different features
        # For simplicity, using same approach
        model = StockPredictionModel(model_type)
        
        X = features_df.drop(columns=['target']).fillna(0).values
        y = features_df['target'].values
        
        # Split with time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            metrics = model.train(X_train, y_train, X_val, y_val)
            all_metrics.append(metrics)
            
            if fold == 4:
                self.long_term_model = model
        
        # Log to MLflow
        with mlflow.start_run(run_name="long_term_model"):
            mlflow.log_params({
                "model_type": model_type,
                "n_features": X.shape[1],
                "n_samples": X.shape[0]
            })
            
            for metric_name in all_metrics[0].keys():
                values = [m[metric_name] for m in all_metrics]
                mlflow.log_metric(f"avg_{metric_name}", np.mean(values))
                mlflow.log_metric(f"std_{metric_name}", np.std(values))
            
            mlflow.sklearn.log_model(model.model, "model")
        
        return all_metrics
    
    def save_models(self, base_path: str = "models/saved_models"):
        """Save trained models"""
        import os
        os.makedirs(base_path, exist_ok=True)
        
        if self.short_term_model:
            self.short_term_model.save(f"{base_path}/short_term_model.pkl")
        
        if self.long_term_model:
            self.long_term_model.save(f"{base_path}/long_term_model.pkl")
        
        logger.info(f"Models saved to {base_path}")