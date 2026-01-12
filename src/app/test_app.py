'''
    Generate Test Data and Test the API

    This module provides functionality to generate test data and test the API.

    Usage:
    python -m src.app.test_app
'''

# ---------- Imports ---------- #
import os
import time
import argparse
import pandas as pd
import requests
from dotenv import load_dotenv
from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('test_app')

# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))

# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
APP_TEST_DATA_FILE = os.getenv('APP_TEST_DATA_FILE', 'app_test_data.csv')

HOST = os.getenv('MAIN_APP_HOST', 'localhost')
PORT = os.getenv('MAIN_APP_VM_PORT', '8080')
API_URL = f"http://{HOST}:{PORT}/predict"

SLEEP_SECONDS = float(os.getenv('SLEEP_SECONDS', 0.5))
MAX_REQUESTS = int(os.getenv('MAX_REQUESTS', 100))

# ---------- Parse Command-Line Arguments ---------- #
parser = argparse.ArgumentParser(description='Load testing for FastAPI service')
parser.add_argument('--limit', type=int, default=MAX_REQUESTS, help='Max requests in limited mode')
parser.add_argument('--sleep', type=float, default=SLEEP_SECONDS, help='Sleep time between requests in seconds')
args = parser.parse_args()

# ---------- Generate Test Data and Save to CSV ---------- #
def generate_test_data():
    '''
        Generates test data and saves it to a CSV file.
    '''
    # TODO: Implement data generation logic
    test_data.to_csv(os.path.join(DATA_DIR, APP_TEST_DATA_FILE), index=False)
    logger.info(f'Test data saved to {os.path.join(DATA_DIR, APP_TEST_DATA_FILE)}')
    return test_data


# ---------- Send Request to API ---------- #
def send_request(i, row):
    '''
        Sends one POST request to the FastAPI prediction API using data from a single row.
        Args:
            customer_id (int) - customer identifier;
            row (pd.Series) - one row from the DataFrame.
    '''
    payload = {
        'customer_id': f'APT_{i}',
        'data': row.to_dict()
    }

    try:
        # Measure request duration
        start_time = time.time()
        response = requests.post(API_URL, json=payload)
        duration = time.time() - start_time

        # Handle the response
        if response.status_code == 200:
            data = response.json()
            logger.info(f"[{i:04d}] Done | price={data['price']:.2f} | request_duration={duration:.3f}s")
        else:
            logger.error(f'[{i:04d}] Failed | response_code={response.status_code} | {response.text[:100]}')

    except requests.exceptions.RequestException as e:
        logger.error(f'[{i:04d}] Request error: {e}')

# ---------- Main ---------- #
if __name__ == '__main__':

    logger.info(f'Running application test for {args.limit} requests')
    generate_test_data()
    for i, (_, row) in enumerate(test_data.head(args.limit).iterrows(), 1):
        send_request(i, row) # i as a customer id of test_data
        time.sleep(args.sleep)
    logger.info('Test finished')
