'''
    Bank Products Recommender API

    This module provides a FastAPI application for bank product recommendations.
    Includes endpoints for health check and single prediction.
    Integrated with Prometheus for metrics collection and monitoring.

    Usage:
    python -m src.api.main_api
    uvicorn src.api.main_api:app --host 0.0.0.0 --port 8080
'''

# ---------- Imports ---------- #
import os
import json
import time
import uvicorn
from typing import List
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from prometheus_client import Counter, Histogram, Gauge, Info
from prometheus_fastapi_instrumentator import Instrumentator

from src.logging_setup import setup_logging
from src.api.get_prediction import ModelHandler, get_model_handler
from src.api.validate_query import QueryValidator
from src.api.schemas import (
    PredictionRequest,
    PredictionResponse,
    ProductRecommendation,
    HealthResponse,
    ErrorResponse,
    ValidationError,
)


# ---------- Logging setup ---------- #
logger = setup_logging('main_api')


# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
if os.path.exists(os.path.join(config_dir, '.env')):
    load_dotenv(os.path.join(config_dir, '.env'))

# API Configuration
RECOMMENDER_HOST = os.getenv('RECOMMENDER_HOST', '0.0.0.0')
RECOMMENDER_PORT = int(os.getenv('RECOMMENDER_PORT', 8080))
API_VERSION = os.getenv('API_VERSION', '1.0.0')


# ---------- Prometheus Metrics ---------- #
# Request counters
REQUEST_COUNTER = Counter(
    'recommender_requests_total',
    'Total number of prediction requests received',
    ['endpoint', 'status']
)

# Prediction latency histogram
PREDICTION_LATENCY = Histogram(
    'recommender_prediction_latency_seconds',
    'Histogram of prediction latency in seconds',
    ['endpoint'],
    buckets=[0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0]
)

# Number of recommendations histogram
RECOMMENDATIONS_COUNT = Histogram(
    'recommender_recommendations_count',
    'Histogram of number of recommendations returned',
    buckets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
)

# Recommendation probability histogram
RECOMMENDATION_PROBABILITY = Histogram(
    'recommender_top_probability',
    'Histogram of top recommendation probability',
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# In-progress requests gauge
INPROGRESS_REQUESTS = Gauge(
    'recommender_inprogress_requests',
    'Number of requests currently being processed',
    ['endpoint']
)

# Validation errors counter
VALIDATION_ERRORS = Counter(
    'recommender_validation_errors_total',
    'Total number of validation errors',
    ['field']
)

# Model info
MODEL_INFO = Info(
    'recommender_model',
    'Information about the loaded model'
)

# Model evaluation metrics (loaded at startup)
MODEL_LOG_LOSS = Gauge(
    'recommender_model_log_loss',
    'Model log loss from evaluation (per product)',
    ['product']
)

MODEL_AUC = Gauge(
    'recommender_model_auc',
    'Model AUC-ROC from evaluation (per product)',
    ['product']
)

MODEL_MEAN_LOG_LOSS = Gauge(
    'recommender_model_mean_log_loss',
    'Mean log loss across all products'
)

MODEL_MEAN_AUC = Gauge(
    'recommender_model_mean_auc',
    'Mean AUC-ROC across all products'
)


# ---------- Global Instances ---------- #
model_handler: ModelHandler = None
validator: QueryValidator = None


# ---------- Lifespan (Startup/Shutdown) ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''
        Application lifespan handler for startup and shutdown.
    '''
    global model_handler, validator
    
    # Startup
    logger.info('Starting Bank Products Recommender API')
    
    try:
        # Initialize model handler
        model_handler = get_model_handler()
        model_handler.load_model()
        
        # Update model info metric
        model_info = model_handler.get_model_info()
        MODEL_INFO.info({
            'version': model_handler.model_version,
            'n_features': str(model_info['n_features']),
            'n_products': str(model_info['n_products']),
        })
        
        # Load and set model evaluation metrics (log loss, AUC)
        results_dir = Path(os.getenv('RESULTS_DIR', './results'))
        eval_files = list(results_dir.glob('ovr_grouped_evaluation_*.json'))
        if eval_files:
            latest_eval_file = sorted(eval_files)[-1]
            with open(latest_eval_file, 'r') as f:
                eval_results = json.load(f)
            
            # Set per-product metrics and calculate means
            per_product = eval_results.get('per_product', {})
            log_losses = []
            aucs = []
            for product, metrics in per_product.items():
                product_name = product.replace('target_', '')
                log_loss_val = metrics.get('log_loss', 0.0)
                auc_val = metrics.get('auc_roc', 0.0)
                
                MODEL_LOG_LOSS.labels(product=product_name).set(log_loss_val)
                MODEL_AUC.labels(product=product_name).set(auc_val)
                
                if log_loss_val > 0:
                    log_losses.append(log_loss_val)
                if auc_val > 0:
                    aucs.append(auc_val)
            
            # Set overall metrics (calculated from per-product)
            overall = eval_results.get('overall', {})
            MODEL_MEAN_AUC.set(overall.get('mean_auc', 0.0))
            MODEL_MEAN_LOG_LOSS.set(sum(log_losses) / len(log_losses) if log_losses else 0.0)
            
            logger.info(f'Loaded evaluation metrics from: {latest_eval_file}')
            logger.info(f'Mean AUC: {overall.get("mean_auc", 0.0):.4f}, Mean Log Loss: {sum(log_losses)/len(log_losses) if log_losses else 0:.4f}')
        else:
            logger.warning('No evaluation results found - model metrics not set')
        
        # Initialize validator
        validator = QueryValidator()
        
        logger.info(f'Model loaded successfully: {model_handler.model_name}')
        logger.info(f'API ready at http://{RECOMMENDER_HOST}:{RECOMMENDER_PORT}')
        
    except Exception as e:
        logger.error(f'Failed to initialize API: {e}')
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info('Shutting down Bank Products Recommender API')


# ---------- FastAPI App ---------- #
app = FastAPI(
    title='Bank Products Recommender API',
    description='''
    ML-powered bank product recommendations API.
    
    This API provides personalized product recommendations for banking customers
    based on their profile and behavior patterns.
    ''',
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc'
)

# Prometheus instrumentation
instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    excluded_handlers=['/health', '/metrics'],
    inprogress_labels=True,
)
instrumentator.instrument(app).expose(app)


# ---------- Exception Handlers ---------- #
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    '''
        Handle HTTP exceptions with custom response format.
    '''
    REQUEST_COUNTER.labels(endpoint=request.url.path, status='error').inc()
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            'error': 'HTTPException',
            'message': exc.detail,
            'details': None
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    '''
        Handle general exceptions.
    '''
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    REQUEST_COUNTER.labels(endpoint=request.url.path, status='error').inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            'error': 'InternalServerError',
            'message': 'An unexpected error occurred'
        }
    )


# ---------- Endpoints ---------- #
@app.get('/', tags=['General'])
async def root():
    '''
        Root endpoint returning API information.
    '''
    return {
        'name': 'Bank Products Recommender API',
        'status': 'running',
        'docs': '/docs',
        'health': '/health',
        'metrics': '/metrics'
    }


@app.get('/health', response_model=HealthResponse, tags=['General'])
async def health_check():
    '''
        Health check endpoint.
    '''
    if model_handler is None:
        return HealthResponse(
            status='unhealthy',
            model_loaded=False,
            version=API_VERSION
        )
    
    health = model_handler.health_check()
    
    return HealthResponse(
        status=health['status'],
        model_loaded=health.get('model_loaded', False),
        version=API_VERSION
    )


@app.get('/model/info', tags=['Model'])
async def get_model_info():
    '''
        Get information about the loaded model.
    '''
    if model_handler is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='Model not loaded'
        )
    
    return model_handler.get_model_info()


@app.post(
    '/predict',
    response_model=PredictionResponse,
    responses={
        400: {'model': ErrorResponse, 'description': 'Validation Error'},
        500: {'model': ErrorResponse, 'description': 'Internal Server Error'}
    },
    tags=['Predictions']
)
async def predict(request: PredictionRequest):
    '''
        Get product recommendations for a single customer.
    '''
    INPROGRESS_REQUESTS.labels(endpoint='/predict').inc()
    start_time = time.time()
    
    try:
        # Convert request to dict for validation
        request_dict = {
            'customer_id': request.customer_id,
            'features': request.features.model_dump()
        }
        
        # Validate request
        is_valid, errors = validator.validate(request_dict)
        
        if not is_valid:
            # Track validation errors
            for err in errors:
                VALIDATION_ERRORS.labels(field=err['field']).inc()
            
            REQUEST_COUNTER.labels(endpoint='/predict', status='validation_error').inc()
            
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    'error': 'ValidationError',
                    'message': 'Request validation failed',
                    'details': errors
                }
            )
        
        # Get predictions
        result = model_handler.predict(
            features=request.features.model_dump(),
            top_k=request.top_k
        )
        
        # Calculate latency
        latency = time.time() - start_time
        latency_ms = latency * 1000
        
        # Record metrics
        PREDICTION_LATENCY.labels(endpoint='/predict').observe(latency)
        RECOMMENDATIONS_COUNT.observe(len(result['recommendations']))
        
        if result['recommendations']:
            top_prob = result['recommendations'][0]['probability']
            RECOMMENDATION_PROBABILITY.observe(top_prob)
        
        REQUEST_COUNTER.labels(endpoint='/predict', status='success').inc()
        
        # Build response
        recommendations = [
            ProductRecommendation(
                product_id=rec['product_id'],
                product_name=rec['product_name'],
                probability=rec['probability'],
                rank=rec['rank']
            )
            for rec in result['recommendations']
        ]
        
        logger.info(f'Prediction completed for customer {request.customer_id}: '
                   f'{len(recommendations)} recommendations, latency={latency_ms:.2f}ms')
        
        return PredictionResponse(
            customer_id=request.customer_id,
            recommendations=recommendations,
            latency_ms=latency_ms,
            model_version=model_handler.model_version
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f'Prediction error for customer {request.customer_id}: {e}')
        REQUEST_COUNTER.labels(endpoint='/predict', status='error').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    finally:
        INPROGRESS_REQUESTS.labels(endpoint='/predict').dec()


# ---------- Main ---------- #
if __name__ == '__main__':
    uvicorn.run(
        'src.api.main_api:app',
        host=RECOMMENDER_HOST,
        port=RECOMMENDER_PORT,
        log_level='info'
    )
