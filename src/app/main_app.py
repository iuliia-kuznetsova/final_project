'''
    Microservice launcher

    This module provides a FastAPI application of bank products recommendation for a customer.
    It includes endpoints for health check, prediction, prediction probabilities, and configuration.

    Usage:
    python -m src.app.main_app
'''

# ---------- Imports ---------- #
from fastapi import FastAPI, HTTPException, Request
import uvicorn
import load_dotenv
import os
from pydantic import BaseModel
import pandas as pd
import polars as pl
import time
import numpy as np

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator

from src.logging_setup import setup_logging
from src.app.preprocess_data import DataHandler
from src.app.get_predictions import ModelHandler
from src.app.validate_query import QueryValidator


# ---------- Logging setup ---------- #
logger = setup_logging('main_services')


# ---------- Load environment variables ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

recommender_port = int(os.getenv('RECOMMENDER_PORT', 8000))


# ---------- Metrics ---------- #
REQUEST_COUNTER = Counter(
    'predictions_total',
    'Total number of prediction requests received'
)

PREDICTION_LATENCY = Histogram(
    'prediction_latency_seconds',
    'Histogram of prediction latency (seconds)'
)

# Histogram of prediction values
PREDICTIONS_HISTOGRAM = Histogram(
    'recommender_predictions',
    'Histogram of predictions'
)

# Gauge for in-progress requests
INPROGRESS_REQUESTS = Gauge(
    'inprogress_requests',
    'Number of requests currently in progress'
)


# ---------- FastAPI App ---------- #
app = FastAPI(
    title='Bank Products Recommender',
    description='ML-powered bank product recommendations',
    version='1.0.0'
)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


# ---------- Model & Validator ---------- #
data_handler = DataHandler()
model_handler = ModelHandler()
validator = QueryValidator()


# ---------- Schema ---------- #
class PredictionRequest(BaseModel):
    apartment_id: str
    model_params: dict


# ---------- Endpoints ---------- #
@app.get('/')
async def root():
    return {'message': 'Bank Products Recommender API'}

@app.get('/health')
async def health_check():
    return {'status': 'healthy'}

@app.post('/predict')
def predict(request: PredictionRequest):
    INPROGRESS_REQUESTS.inc()
    start_time = time.time()
    REQUEST_COUNTER.inc()

    try:
        params = request.dict()
        if not validator.validate(params):
            raise HTTPException(status_code=400, detail='Invalid request parameters')

        X = data_handler.prepare_data()
        recommendations = model_handler.recommend(X)

        PREDICTIONS_HISTOGRAM.observe(recommendations)

        latency = time.time() - start_time
        PREDICTION_LATENCY.observe(latency)

        # TODO: Check output format of recommendations
        logger.info(f'Recommendations: {recommendations}')
        return {'customer_id': customer_id, 'bank_product_id': recommendations, 'latency_sec': round(latency, 4)}
    finally:
        INPROGRESS_REQUESTS.dec()

# ---------- Main ---------- #
if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=recommender_port)
