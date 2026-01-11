'''
    Data Preprocessing

    This module provides functionality to preprocess raw test data 
    into a format suitable for model prediction.

    Usage:
    python -m src.app.preprocess_data
'''

# ---------- Imports ---------- #
import os
import polars as pl
import pandas as pd
from dotenv import load_dotenv
import numpy as np
from typing import Dict, Any

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('preprocess_data')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
APP_TEST_DATA_FILE = os.getenv('APP_TEST_DATA_FILE', 'app_test_data.parquet')


# ---------- Data Handler ---------- #
class DataHandler:
    '''
        Data Handler class.
        This class provides functionality to load and prepare test data for model prediction.
    '''
    def __init__(self):
        self.test_data = pd.read_parquet(os.path.join(DATA_DIR, APP_TEST_DATA_FILE))
        logger.info(f'Test data loaded from {os.path.join(DATA_DIR, APP_TEST_DATA_FILE)}')
    
    def prepare_data(self):
        # TODO: Implement data preparation logic
        
        self.test_data.to_csv(os.path.join(DATA_DIR, 'app_test_data_preprocessed.csv'), index=False)
        logger.info(f'Test data prepared and saved to {os.path.join(DATA_DIR, "app_test_data_preprocessed.csv")}')

        return None

# ---------- Main ---------- #
if __name__ == '__main__':

    logger.info('Starting data preparation')
    data_handler = DataHandler()
    data_handler.prepare_data()
    logger.info('Data preparation completed')

# ---------- All Exports ---------- #
__all__ = ['DataHandler']