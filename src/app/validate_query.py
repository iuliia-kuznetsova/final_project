'''
    Query Validator

    This module provides functionality to validate API request parameters.

    Usage:
    python -m src.app.validate_query
'''

# ---------- Imports ---------- #
import os
from dotenv import load_dotenv
from typing import Dict, Any

from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('query_validator')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

REQUIRED_DATA = [
    ... # TODO: Add required data
]

# ---------- Query Validator ---------- #
class QueryValidator:
    '''
        Query Validator class.
        This class provides functionality to validate API request data.
    '''

    def __init__(self, required_data: list):
        self.required_data = required_data
        self.data_types = {
            ... # TODO: Add required data types
        }

    def check_required_query_params(self, query_params: dict) -> bool:
        if 'customer_id' not in query_params or 'data' not in query_params:
            return False

        if not isinstance(query_params['customer_id'], self.data_types['customer_id']):
            return False

        # TODO: Add check for all other features
        logger.info(f"Query parameters checked: {query_params}")
        return True

    def check_required_data(self, data: dict) -> bool:
        # TODO: Add check
        logger.info(f"Data checked: {data}")
        return set(self.required_data).issubset(data.keys())

    def validate(self, params: dict) -> bool:
        # TODO: Add check
        if not self.check_required_query_params(params):
            logger.error(f"Missing or invalid 'customer_id' or 'data': {params}")
            return False
        if not self.check_required_data(params['data']):
            logger.error(f"Missing required data: {params['data']}")
            return False
        logger.info(f"Query parameters validated: {params}")
        return True

# ---------- Main ---------- #
if __name__ == '__main__':
    validator = QueryValidator(REQUIRED_DATA)