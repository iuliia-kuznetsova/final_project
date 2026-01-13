'''
    Bank Products Recommender API

    This module provides a FastAPI application for bank product recommendations.
    Includes endpoints for health check, single prediction, and batch predictions.
    Integrated with Prometheus for metrics collection and monitoring.

    Usage:
    python -m src.api.main_api
    or
    uvicorn src.api.main_api:app --host 0.0.0.0 --port 8080
'''

# ---------- Imports ---------- #
import os
import time
import uvicorn
from typing import List
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
    BatchPredictionRequest,
    PredictionResponse,
    BatchPredictionResponse,
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
DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'


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

# Batch size histogram
BATCH_SIZE = Histogram(
    'recommender_batch_size',
    'Histogram of batch prediction sizes',
    buckets=[1, 5, 10, 25, 50, 100, 250, 500, 1000]
)

# Model info
MODEL_INFO = Info(
    'recommender_model',
    'Information about the loaded model'
)


# ---------- Global Instances ---------- #
model_handler: ModelHandler = None
validator: QueryValidator = None


# ---------- Lifespan (Startup/Shutdown) ---------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    '''Application lifespan handler for startup and shutdown.'''
    global model_handler, validator
    
    # Startup
    logger.info('Starting Bank Products Recommender API...')
    
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
        
        # Initialize validator
        validator = QueryValidator()
        
        logger.info(f'Model loaded successfully: {model_handler.model_name}')
        logger.info(f'API ready at http://{RECOMMENDER_HOST}:{RECOMMENDER_PORT}')
        
    except Exception as e:
        logger.error(f'Failed to initialize API: {e}')
        raise
    
    yield  # Application is running
    
    # Shutdown
    logger.info('Shutting down Bank Products Recommender API...')


# ---------- FastAPI App ---------- #
app = FastAPI(
    title='Bank Products Recommender API',
    description='''
    ML-powered bank product recommendations API.
    
    This API provides personalized product recommendations for banking customers
    based on their profile and behavior patterns.
    
    ## Features
    - **Single Prediction**: Get top-K recommendations for a single customer
    - **Batch Prediction**: Get recommendations for multiple customers at once
    - **Prometheus Metrics**: Built-in metrics for monitoring and alerting
    
    ## Model
    Uses a One-vs-Rest (OvR) CatBoost model trained on customer data to predict
    the probability of product acquisition.
    ''',
    version=API_VERSION,
    lifespan=lifespan,
    docs_url='/docs',
    redoc_url='/redoc'
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
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
    '''Handle HTTP exceptions with custom response format.'''
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
    '''Handle general exceptions.'''
    logger.error(f'Unhandled exception: {exc}', exc_info=True)
    REQUEST_COUNTER.labels(endpoint=request.url.path, status='error').inc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            'error': 'InternalServerError',
            'message': 'An unexpected error occurred',
            'details': [{'field': 'server', 'message': str(exc)}] if DEBUG_MODE else None
        }
    )


# ---------- Endpoints ---------- #
@app.get('/', tags=['General'])
async def root():
    '''Root endpoint returning API information.'''
    return {
        'name': 'Bank Products Recommender API',
        'version': API_VERSION,
        'status': 'running',
        'docs': '/docs',
        'health': '/health',
        'metrics': '/metrics'
    }


@app.get('/health', response_model=HealthResponse, tags=['General'])
async def health_check():
    '''
    Health check endpoint.
    
    Returns the current health status of the API and model.
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
    
    Returns model metadata including version, number of features, and product groups.
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
    
    Accepts customer features and returns top-K product recommendations
    with probabilities.
    
    - **customer_id**: Unique customer identifier
    - **features**: Customer features for prediction
    - **top_k**: Number of recommendations to return (default: 7, max: 24)
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


@app.post(
    '/predict/batch',
    response_model=BatchPredictionResponse,
    responses={
        400: {'model': ErrorResponse, 'description': 'Validation Error'},
        500: {'model': ErrorResponse, 'description': 'Internal Server Error'}
    },
    tags=['Predictions']
)
async def predict_batch(request: BatchPredictionRequest):
    '''
    Get product recommendations for multiple customers.
    
    Accepts a list of customer features and returns recommendations for each.
    Maximum batch size is 1000 customers.
    
    - **customers**: List of prediction requests
    '''
    INPROGRESS_REQUESTS.labels(endpoint='/predict/batch').inc()
    start_time = time.time()
    
    try:
        # Validate batch size
        if len(request.customers) > 1000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail='Batch size exceeds maximum of 1000'
            )
        
        BATCH_SIZE.observe(len(request.customers))
        
        # Validate all requests
        all_errors = {}
        for idx, customer_request in enumerate(request.customers):
            request_dict = {
                'customer_id': customer_request.customer_id,
                'features': customer_request.features.model_dump()
            }
            is_valid, errors = validator.validate(request_dict)
            if not is_valid:
                all_errors[idx] = errors
        
        if all_errors:
            REQUEST_COUNTER.labels(endpoint='/predict/batch', status='validation_error').inc()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={
                    'error': 'BatchValidationError',
                    'message': f'Validation failed for {len(all_errors)} requests',
                    'details': all_errors
                }
            )
        
        # Prepare batch features
        batch_features = [req.features.model_dump() for req in request.customers]
        top_k_values = [req.top_k for req in request.customers]
        
        # Get batch predictions
        results = model_handler.predict_batch(
            batch_features=batch_features,
            top_k=max(top_k_values)  # Use max top_k for batch
        )
        
        # Calculate latency
        latency = time.time() - start_time
        latency_ms = latency * 1000
        
        # Record metrics
        PREDICTION_LATENCY.labels(endpoint='/predict/batch').observe(latency)
        REQUEST_COUNTER.labels(endpoint='/predict/batch', status='success').inc()
        
        # Build responses
        predictions = []
        for idx, (customer_request, result) in enumerate(zip(request.customers, results)):
            # Trim to requested top_k
            recs = result['recommendations'][:customer_request.top_k]
            
            recommendations = [
                ProductRecommendation(
                    product_id=rec['product_id'],
                    product_name=rec['product_name'],
                    probability=rec['probability'],
                    rank=rec['rank']
                )
                for rec in recs
            ]
            
            predictions.append(PredictionResponse(
                customer_id=customer_request.customer_id,
                recommendations=recommendations,
                latency_ms=latency_ms / len(request.customers),  # Average per customer
                model_version=model_handler.model_version
            ))
        
        logger.info(f'Batch prediction completed: {len(predictions)} customers, '
                   f'total_latency={latency_ms:.2f}ms')
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_latency_ms=latency_ms,
            batch_size=len(predictions)
        )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f'Batch prediction error: {e}')
        REQUEST_COUNTER.labels(endpoint='/predict/batch', status='error').inc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    
    finally:
        INPROGRESS_REQUESTS.labels(endpoint='/predict/batch').dec()


# ---------- Main ---------- #
if __name__ == '__main__':
    uvicorn.run(
        'src.api.main_api:app',
        host=RECOMMENDER_HOST,
        port=RECOMMENDER_PORT,
        reload=DEBUG_MODE,
        log_level='info'
    )
