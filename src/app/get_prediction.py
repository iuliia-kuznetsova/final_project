'''
    Model Handler

    This module provides functionality to load and use the OvR model for prediction.

    Usage:
    python -m src.app.get_prediction
'''

# ---------- Imports ---------- #
import os
import polars as pl
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from typing import Dict, Any

from src.logging_setup import setup_logging
from src.recs.modelling_ovr import OvRGroupModel


# ---------- Logging setup ---------- #
logger = setup_logging('model_handler')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
APP_TEST_DATA_FILE = os.getenv('APP_TEST_DATA_FILE', 'app_test_data_preprocessed.csv')
MODEL_DIR = os.getenv('MODEL_DIR', './models')
MODEL_NAME = os.getenv('MODEL_NAME', 'ovr_grouped_catboost')


# ---------- Model Handler ---------- #
class ModelHandler:
    '''
        Model Handler class.
        This class provides functionality to load and use the OvR model for prediction.
    '''
    def __init__(self):
        self.model = OvRGroupModel()

    def load_model(self):
        self.model.load(os.path.join(MODEL_DIR, MODEL_NAME))
        logger.info(f'Model loaded from {os.path.join(MODEL_DIR, MODEL_NAME)}')
        return self.model

    def load_data(self):
        self.data = pd.read_csv(os.path.join(DATA_DIR, 'app_test_data_preprocessed.csv'))
        logger.info(f'Data loaded from {os.path.join(DATA_DIR, 'app_test_data_preprocessed.csv')}')
        return self.data

    def recommend(self) -> pd.DataFrame:
        recommendations = self.model.recommend(self.data)
        logger.info(f'Recommendations: {recommendations}')
        return recommendations

# ---------- Main ---------- #
if __name__ == '__main__':
    
    logger.info('Starting prediction')
    model_handler = ModelHandler()
    model = model_handler.load_model()
    data = model_handler.load_data()
    recommendations = model.recommend(data)
    logger.info(f'Recommendations: {recommendations}')
    logger.info('Prediction completed')