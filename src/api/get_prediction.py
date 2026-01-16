'''
    Model Handler

    This module provides functionality to load and use the OvR model for prediction.
    Handles model loading, data preprocessing, and generating top-K recommendations.

    Usage:
    from src.api.get_prediction import ModelHandler
    handler = ModelHandler()
    handler.load_model()
    recommendations = handler.predict(features_dict)
'''

# ---------- Imports ---------- #
import os
import json
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Tuple, Optional, Union

from src.logging_setup import setup_logging
from src.recs.modelling_ovr import OvRGroupModel
from src.api.schemas import PRODUCT_NAMES


# ---------- Logging setup ---------- #
logger = setup_logging('model_handler')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
MODEL_DIR = os.getenv('MODEL_DIR', './models')
MODEL_NAME = os.getenv('MODEL_NAME', 'ovr_grouped_santander')
TOP_K = int(os.getenv('TOP_K', 7))

# Product name mapping for human-readable names
PRODUCT_NAME_MAPPING = {
    'target_ahor_fin': 'Saving Account',
    'target_aval_fin': 'Guarantees',
    'target_cco_fin': 'Current Account',
    'target_cder_fin': 'Derivada Account',
    'target_cno_fin': 'Payroll Account',
    'target_ctju_fin': 'Junior Account',
    'target_ctma_fin': 'Más Particular Account',
    'target_ctop_fin': 'Particular Account',
    'target_ctpp_fin': 'Particular Plus Account',
    'target_deco_fin': 'Short-term Deposits',
    'target_deme_fin': 'Medium-term Deposits',
    'target_dela_fin': 'Long-term Deposits',
    'target_ecue_fin': 'e-Account',
    'target_fond_fin': 'Funds',
    'target_hip_fin': 'Mortgage',
    'target_plan_fin': 'Pension Plan',
    'target_pres_fin': 'Loans',
    'target_reca_fin': 'Taxes',
    'target_tjcr_fin': 'Credit Card',
    'target_valo_fin': 'Securities',
    'target_viv_fin': 'Home Account',
    'target_nomina': 'Payroll',
    'target_nom_pens': 'Pension Payroll',
    'target_recibo': 'Direct Debit',
}


# ---------- Model Handler ---------- #
class ModelHandler:
    '''
        Model Handler class.
        This class provides functionality to load and use the OvR model for prediction.
        
        Features:
        - Lazy loading of model (load on first prediction)
        - Feature preprocessing for inference
        - Top-K recommendations with probabilities
        - Thread-safe model access
    '''
    
    def __init__(
        self,
        model_dir: str = MODEL_DIR,
        model_name: str = MODEL_NAME,
        data_dir: str = DATA_DIR,
        top_k: int = TOP_K
    ):
        '''
            Initialize the model handler.
        '''
        self.model_dir = Path(model_dir)
        self.model_name = model_name
        self.data_dir = Path(data_dir)
        self.top_k = top_k
        
        # Model state
        self.model: Optional[OvRGroupModel] = None
        self.is_loaded = False
        self.model_version = 'unknown'
        self.feature_names: List[str] = []
        self.cat_features: List[str] = []
        
        logger.info(f'ModelHandler initialized: model_dir={model_dir}, model_name={model_name}')

    def load_model(self) -> 'ModelHandler':
        '''
            Load the trained OvR model from disk.
        '''
        model_path = self.model_dir / self.model_name
        
        if not model_path.exists():
            raise FileNotFoundError(f'Model not found at: {model_path}')
        
        logger.info(f'Loading model from: {model_path}')
        
        # Initialize and load model (disable MLflow for inference)
        self.model = OvRGroupModel(mlflow_experiment=None)
        self.model.load(str(model_path))
        
        # Extract model metadata
        self.feature_names = self.model.feature_names
        self.cat_features = self.model.cat_features
        self.is_loaded = True
        
        # Set model version from metadata or timestamp
        metadata_path = model_path / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.model_version = metadata.get('model_version', 
                    datetime.now().strftime('%Y%m%d'))
        else:
            self.model_version = datetime.now().strftime('%Y%m%d')
        
        logger.info(f'Model loaded successfully: version={self.model_version}, '
                   f'features={len(self.feature_names)}, cat_features={len(self.cat_features)}')
        
        return self

    def _ensure_loaded(self):
        '''
            Ensure model is loaded before prediction.
        '''
        if not self.is_loaded or self.model is None:
            self.load_model()

    def _preprocess_features(
        self, 
        features: Dict[str, Any]
    ) -> pd.DataFrame:
        '''
            Preprocess features for model inference.
        '''
        # Convert dict to DataFrame
        df = pd.DataFrame([features])
        
        # Get expected feature columns (excluding identifiers and date)
        id_cols = ['fecha_dato', 'ncodpers']
        model_features = [f for f in self.feature_names if f not in id_cols]
        
        # Ensure all required features are present
        for col in model_features:
            if col not in df.columns:
                # Add default value based on feature type
                if col in self.cat_features:
                    df[col] = 'missing'
                elif 'lag' in col or 'interaction' in col or 'acquired' in col:
                    df[col] = 0
                elif col.startswith('ind_'):
                    df[col] = False
                else:
                    df[col] = 0
        
        # Select only model features in correct order
        df = df[model_features]
        
        # Convert categorical columns to string
        for col in self.cat_features:
            if col in df.columns:
                df[col] = df[col].astype(str).replace({'nan': 'missing', 'None': 'missing'})
        
        # Convert boolean columns
        bool_cols = [c for c in df.columns if c.startswith('ind_') and 'interaction' not in c and 'acquired' not in c]
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        logger.debug(f'Preprocessed features: {df.shape}')
        
        return df

    def predict(
        self, 
        features: Dict[str, Any],
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        '''
            Generate top-K product recommendations for a customer.
        '''
        self._ensure_loaded()
        
        k = top_k or self.top_k
        
        # Preprocess features
        X = self._preprocess_features(features)
        
        # Get predictions
        top_k_indices, top_k_probas, top_k_names = self.model.predict_top_k(X, k=k)
        
        # Build recommendations list
        recommendations = []
        for i, (product_id, prob) in enumerate(zip(top_k_names[0], top_k_probas[0])):
            recommendations.append({
                'product_id': product_id,
                'product_name': PRODUCT_NAME_MAPPING.get(product_id, product_id),
                'probability': float(prob),
                'rank': i + 1
            })
        
        logger.info(f'Generated {len(recommendations)} recommendations')
        
        return {
            'recommendations': recommendations,
            'top_k': k
        }

    def predict_batch(
        self, 
        batch_features: List[Dict[str, Any]],
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        '''
            Generate recommendations for a batch of customers.
        '''
        self._ensure_loaded()
        
        k = top_k or self.top_k
        
        # Preprocess all features
        dfs = [self._preprocess_features(f) for f in batch_features]
        X = pd.concat(dfs, ignore_index=True)
        
        # Get predictions
        top_k_indices, top_k_probas, top_k_names = self.model.predict_top_k(X, k=k)
        
        # Build recommendations for each customer
        results = []
        for customer_idx in range(len(batch_features)):
            recommendations = []
            for rank, (product_id, prob) in enumerate(
                zip(top_k_names[customer_idx], top_k_probas[customer_idx])
            ):
                recommendations.append({
                    'product_id': product_id,
                    'product_name': PRODUCT_NAME_MAPPING.get(product_id, product_id),
                    'probability': float(prob),
                    'rank': rank + 1
                })
            
            results.append({
                'recommendations': recommendations,
                'top_k': k
            })
        
        logger.info(f'Generated recommendations for {len(results)} customers')
        
        return results

    def get_model_info(self) -> Dict[str, Any]:
        '''
            Get model metadata and information.
        '''
        self._ensure_loaded()
        
        return {
            'model_name': self.model_name,
            'model_version': self.model_version,
            'n_features': len(self.feature_names),
            'n_cat_features': len(self.cat_features),
            'n_products': len(self.model.all_products),
            'groups': {
                group: len(products) 
                for group, products in self.model.product_groups.items()
            },
            'top_k_default': self.top_k,
            'is_loaded': self.is_loaded
        }

    def health_check(self) -> Dict[str, Any]:
        '''
            Perform health check on the model.
        '''
        try:
            self._ensure_loaded()
            
            # Quick validation - create dummy prediction
            dummy_features = {col: 0 for col in self.feature_names}
            for col in self.cat_features:
                dummy_features[col] = 'test'
            
            # This should not raise an exception
            _ = self._preprocess_features(dummy_features)
            
            return {
                'status': 'healthy',
                'model_loaded': True,
                'model_version': self.model_version
            }
        except Exception as e:
            logger.error(f'Health check failed: {e}')
            return {
                'status': 'unhealthy',
                'model_loaded': False,
                'error': str(e)
            }


# Load model once at startup, every /predict endpoint reuses it
_model_handler: Optional[ModelHandler] = None


def get_model_handler() -> ModelHandler:
    '''
        Get or create the global ModelHandler instance.
    '''
    global _model_handler
    if _model_handler is None:
        _model_handler = ModelHandler()
    return _model_handler


# ---------- Main ---------- #
if __name__ == '__main__':
    
    logger.info('Testing ModelHandler')
    
    # Create handler and load model
    handler = ModelHandler()
    
    try:
        handler.load_model()
        
        # Get model info
        info = handler.get_model_info()
        logger.info(f'Model info: {info}')
        
        # Health check
        health = handler.health_check()
        logger.info(f'Health check: {health}')
        
        # Test prediction with dummy data
        dummy_features = {
            'fecha_dato': '2016-05-28',
            'ncodpers': '12345678',
            'ind_empleado': 'A',
            'pais_residencia': 'ES',
            'sexo': 'H',
            'age': 35,
            'ind_nuevo': False,
            'indrel': '1',
            'customer_period': 12,
            'renta': 50000.0,
        }
        
        result = handler.predict(dummy_features)
        logger.info(f'Recommendations: {result}')
        
    except FileNotFoundError as e:
        logger.error(f'Model not found: {e}')
    except Exception as e:
        logger.error(f'Error: {e}')
