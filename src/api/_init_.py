'''
    Bank Products Recommender API Module

    This module provides a FastAPI-based recommendation service with:
    - Main API module for the recommendation service
    - Prometheus metrics collection
    - Grafana dashboards for monitoring
'''

from src.api.schemas import (
    FEATURE_SCHEMA,
    REQUIRED_FEATURES,
    PRODUCT_NAMES,
    CustomerFeatures,
    PredictionRequest,
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
    ProductRecommendation,
    HealthResponse,
    ErrorResponse,
)

from src.api.validate_query import QueryValidator
from src.api.get_prediction import ModelHandler, get_model_handler

__all__ = [
    # Schemas
    'FEATURE_SCHEMA',
    'REQUIRED_FEATURES',
    'PRODUCT_NAMES',
    'CustomerFeatures',
    'PredictionRequest',
    'BatchPredictionRequest',
    'PredictionResponse',
    'BatchPredictionResponse',
    'ProductRecommendation',
    'HealthResponse',
    'ErrorResponse',
    # Handlers
    'QueryValidator',
    'ModelHandler',
    'get_model_handler',
]

