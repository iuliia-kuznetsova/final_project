'''
    API Testing Module

    This module provides functionality to:
    - Sample data from data_preprocessed.parquet
    - Send requests to the recommender API
    - Validate responses and collect metrics

    Usage:
    python -m src.api.test_api --limit 100 --sleep 0.1
'''

# ---------- Imports ---------- #
import os
import gc
import time
import json
import argparse
import requests
import pandas as pd
import polars as pl
from datetime import datetime
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from src.logging_setup import setup_logging
from src.api.schemas import REQUIRED_FEATURES


# ---------- Logging setup ---------- #
logger = setup_logging('test_api')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
API_TEST_DATA_FILE = os.getenv('API_TEST_DATA_FILE', 'api_test_data.parquet')
API_SAMPLE_NUMBER = int(os.getenv('API_SAMPLE_NUMBER', 1000))

HOST = os.getenv('RECOMMENDER_HOST', 'localhost')
PORT = os.getenv('RECOMMENDER_PORT', '8080')
API_BASE_URL = f"http://{HOST}:{PORT}"
PREDICT_URL = f"{API_BASE_URL}/predict"
HEALTH_URL = f"{API_BASE_URL}/health"

APP_SLEEP_SECONDS = float(os.getenv('APP_SLEEP_SECONDS', 0.1))
APP_REQUESTS_NUMBER = int(os.getenv('APP_REQUESTS_NUMBER', 100))


# ---------- Parse Command-Line Arguments ---------- #
def parse_args():
    parser = argparse.ArgumentParser(description='Load testing for Bank Products Recommender API')
    parser.add_argument('--limit', type=int, default=APP_REQUESTS_NUMBER, 
                       help='Number of requests to send (max 1000)')
    parser.add_argument('--sleep', type=float, default=APP_SLEEP_SECONDS, 
                       help='Sleep time between requests in seconds')
    parser.add_argument('--host', type=str, default=HOST, 
                       help='API host')
    parser.add_argument('--port', type=int, default=int(PORT), 
                       help='API port')
    parser.add_argument('--sample', action='store_true', 
                       help='Generate new sample data before testing')
    parser.add_argument('--top-k', type=int, default=7, 
                       help='Number of recommendations to request')
    return parser.parse_args()


# ---------- Data Sampling ---------- #
def sample_data(
    data_dir: str = DATA_DIR,
    preprocessed_file: str = PREPROCESSED_DATA_FILE,
    output_file: str = API_TEST_DATA_FILE,
    sample_size: int = API_SAMPLE_NUMBER,
    random_state: int = 42
) -> pd.DataFrame:
    '''
        Sample data from preprocessed parquet file.
    '''
    
    # Load data with Polars (faster for large files)
    data_path = os.path.join(data_dir, preprocessed_file)
    df = pl.read_parquet(data_path)
    
    logger.info(f'Preprocessed data loaded: {df.height:,} rows x {df.width} columns')
    
    # Sample data
    if sample_size >= df.height:
        sample_df = df
        logger.info(f'Sample size >= data size, using all {df.height:,} rows')
    else:
        sample_df = df.sample(n=sample_size, seed=random_state)
        logger.info(f'Data sampled: {sample_size:,} rows')
    
    # Drop target columns (not needed for prediction)
    target_cols = [col for col in sample_df.columns if col.startswith('target_')]
    sample_df = sample_df.drop(target_cols)
    
    logger.info(f'Target columns for API testing dropped: {len(target_cols)}')
    
    # Save sample to parquet
    output_path = os.path.join(data_dir, output_file)
    sample_df.write_parquet(output_path)
    logger.info(f'Sampled data saved to: {output_path}')
    
    # Convert to pandas for API testing
    sample_pd = sample_df.to_pandas()
    logger.info(f'Data converted to pandas for API testing')
    logger.info(f'DONE: Sampling completed')

    # Clean up
    del df, sample_df
    gc.collect()
    
    return sample_pd


def load_test_data(
    data_dir: str = DATA_DIR,
    test_file: str = API_TEST_DATA_FILE
) -> pd.DataFrame:
    '''
        Load existing test data from parquet file.
    '''
    data_path = os.path.join(data_dir, test_file)
    
    if not os.path.exists(data_path):
        logger.warning(f'Api test data file not found: {data_path}')
        return sample_data(data_dir)
    
    df = pl.read_parquet(data_path).to_pandas()
    logger.info(f'Api test data loaded: {len(df):,} rows')
    logger.info(f'DONE: Api test data loaded')
    return df


# ---------- Data Conversion ---------- #
def row_to_request(
    row: pd.Series,
    top_k: int = 7
) -> Dict[str, Any]:
    '''
        Convert a DataFrame row to API request format.
    '''
    # Extract customer ID
    customer_id = str(row.get('ncodpers', 'unknown'))
    
    # Convert row to features dict
    features = {}
    for col in row.index:
        value = row[col]
        
        # Handle different types
        if pd.isna(value):
            # Skip NA values, will use defaults
            continue
        elif isinstance(value, (pd.Timestamp, datetime)):
            features[col] = value.strftime('%Y-%m-%d')
        elif hasattr(value, 'item'):  # numpy types
            features[col] = value.item()
        else:
            features[col] = value
    
    return {
        'customer_id': customer_id,
        'features': features,
        'top_k': top_k
    }


# ---------- API Client Functions ---------- #
def check_health(base_url: str = API_BASE_URL) -> bool:
    '''
        Check if the API is healthy.
    '''
    try:
        response = requests.get(f'{base_url}/health', timeout=5)
        if response.status_code == 200:
            health = response.json()
            logger.info(f"API health check: {health['status']}, model_loaded={health.get('model_loaded', False)}")
            return health.get('status') == 'healthy'
        else:
            logger.error(f'Health check failed: {response.status_code}')
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f'Health check error: {e}')
        return False


def send_single_request(
    request_data: Dict[str, Any],
    base_url: str = API_BASE_URL,
    request_idx: int = 0
) -> Optional[Dict[str, Any]]:
    '''
        Send a single prediction request to the API.
    '''
    try:
        start_time = time.time()
        response = requests.post(
            f'{base_url}/predict',
            json=request_data,
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            customer_id = result.get('customer_id', 'unknown')
            n_recs = len(result.get('recommendations', []))
            latency_ms = result.get('latency_ms', elapsed * 1000)
            
            # Get top recommendation
            top_rec = None
            if result.get('recommendations'):
                top_rec = result['recommendations'][0]
            
            logger.info(
                f"Request [{request_idx:04d}] successfully completed | "
                f"customer id: {customer_id} | "
                f"number of recommendations: {n_recs} latency: {latency_ms:.1f}ms | "
                f"top recommendation: {top_rec['product_id'] if top_rec else 'N/A'} with probability: {top_rec['probability']:.3f}"
            )
            return result
        else:
            error_detail = response.text[:200]
            logger.error(
                f"Request [{request_idx:04d}] failed with status={response.status_code} | "
                f"and error_detail: {error_detail}"
            )
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Request [{request_idx:04d}] failed with error: {e}")
        return None


# ---------- Test Runners ---------- #
def run_single_prediction_test(
    test_data: pd.DataFrame,
    limit: int = 100,
    sleep_seconds: float = 0.1,
    top_k: int = 7,
    base_url: str = API_BASE_URL
) -> Dict[str, Any]:
    '''
        Run single prediction test.
    '''
    logger.info(f'Running single prediction test: {limit} requests, sleep: {sleep_seconds}s')
    
    results = {
        'total_requests': 0,
        'successful': 0,
        'failed': 0,
        'latencies': [],
        'start_time': datetime.now().isoformat()
    }
    
    # Limit test data
    test_subset = test_data.head(limit)
    
    for idx, (_, row) in enumerate(test_subset.iterrows()):
        request_data = row_to_request(row, top_k=top_k)
        response = send_single_request(request_data, base_url, idx)
        
        results['total_requests'] += 1
        
        if response:
            results['successful'] += 1
            results['latencies'].append(response.get('latency_ms', 0))
        else:
            results['failed'] += 1
        
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
    
    # Calculate statistics
    results['end_time'] = datetime.now().isoformat()
    results['success_rate'] = results['successful'] / results['total_requests'] if results['total_requests'] > 0 else 0
    
    if results['latencies']:
        import numpy as np
        results['latency_stats'] = {
            'mean': np.mean(results['latencies']),
            'median': np.median(results['latencies']),
            'p95': np.percentile(results['latencies'], 95),
            'p99': np.percentile(results['latencies'], 99),
            'min': np.min(results['latencies']),
            'max': np.max(results['latencies'])
        }
    
    logger.info(f"Single prediction test completed: {results['successful']} successful requests out of {results['total_requests']}")
    
    return results


# ---------- Main ---------- #
def main():
    args = parse_args()
    
    # Validate args
    if args.limit > 1000:
        logger.error('Limit cannot be greater than 1000')
        return
    
    # Update URLs with command line args
    global API_BASE_URL, PREDICT_URL, HEALTH_URL
    API_BASE_URL = f"http://{args.host}:{args.port}"
    PREDICT_URL = f"{API_BASE_URL}/predict"
    HEALTH_URL = f"{API_BASE_URL}/health"
    
    logger.info(f'Starting API Testing')
    
    # Check API health first
    if not check_health(API_BASE_URL):
        logger.error('API is not healthy')
        return
    
    # Load or sample test data
    if args.sample:
        test_data = sample_data(DATA_DIR)
    else:
        test_data = load_test_data(DATA_DIR)
    
    # Run single prediction tests
    logger.info('Starting Single Prediction Tests')
    
    single_results = run_single_prediction_test(
        test_data=test_data,
        limit=args.limit,
        sleep_seconds=args.sleep,
        top_k=args.top_k,
        base_url=API_BASE_URL
    )
    
    # Print summary
    logger.info('Test Summary')
    
    logger.info(f"Single Prediction Results: | "
                f"Total requests: {single_results['total_requests']} | "
                f"Successful requests: {single_results['successful']} | "
                f"Failed requests: {single_results['failed']} | "
                f"Success rate: {single_results['success_rate']:.1%} | "
            )
    
    if 'latency_stats' in single_results:
        stats = single_results['latency_stats']
        logger.info(f"Latency (ms): | "
                    f"Mean: {stats['mean']:.1f} | "
                    f"Median: {stats['median']:.1f} | "
                    f"P95: {stats['p95']:.1f} | "
                    f"P99: {stats['p99']:.1f} | "
                    f"Min: {stats['min']:.1f} | "
                    f"Max: {stats['max']:.1f}"
                )
    
    # Save results to file
    results_file = os.path.join(DATA_DIR, 'api_test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'single_prediction': single_results,
            'config': {
                'host': args.host,
                'port': args.port,
                'limit': args.limit,
                'top_k': args.top_k
            }
        }, f, indent=2, default=str)
    
    logger.info(f'Results saved to: {results_file}')
    logger.info('API testing successfully completed')


# ---------- Main ---------- #
if __name__ == '__main__':
    main()


# ---------- Main ---------- #
if __name__ == '__main__':
    main()
