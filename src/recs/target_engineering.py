'''
    Target Engineering

    This module provides functionality to engineer targets from preprocessed data.

    Input:
    - data_preprocessed - Preprocessed data
    - results_dir - Output directory for results parquet files.

    Output:
    - data_preprocessed.parquet - Data file after raw data preprocessing and target engineering

    Usage:
    python -m src.recs.target_engineering
'''

# ---------- Imports ---------- #
import os
import gc
import polars as pl
from dotenv import load_dotenv
from pathlib import Path

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('target_engineering')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
# Preprocessed data summary file
PREPROCESSED_DATA_SUMMARY_FILE = os.getenv('PREPROCESSED_DATA_SUMMARY_FILE', 'data_preprocessed_summary.parquet')


# ---------- Functions ---------- #
def engineer_targets( 
    data_dir: str = DATA_DIR, 
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    preprocessed_data_summary_file: str = PREPROCESSED_DATA_SUMMARY_FILE
) -> pl.DataFrame:
    '''
        Engineer targets from preprocessed data.    
        
        Strategy:
        - Create targets
        - Filter out rows from the last month in the dataset
    '''

    # Load preprocessed data
    df_preprocessed = pl.read_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f'Loaded preprocessed data from: {data_dir}/{preprocessed_data_file}')
    
    # Create targets
    # Binary targets: a target column for each of 24 products 
    # that is 1 if the customer doesn't have the product now but will have it next month
    # (i.e., predicting product additions)
    # Get all products
    products = [col for col in df_preprocessed.columns if col.startswith('ind_') and col.endswith('_ult1')]
    # Sort by customer and date
    df_engineered = df_preprocessed.sort(['ncodpers', 'fecha_dato'])
    # Create targets using proper Polars syntax with .over()
    target_cols = []
    for prod in products:
        target_name = f"target_{prod.replace('ind_', '').replace('_ult1', '')}"
        target = (
            (pl.col(prod).shift(-1).over('ncodpers') == 1) &  # Customer will have product next month
            (pl.col(prod) == 0)  # Customer doesn't have product now
        ).alias(target_name)
        target_cols.append(target)
    # Add targets to the dataframe
    df_engineered = df_engineered.with_columns(target_cols)
    logger.info(f'Targets created: {[col.meta.output_name() for col in target_cols]}')

    # Filter out rows from the last month in the dataset
    # Since shift(-1) looks at next month, the last month has no valid targets
    # Get target column names
    target_names = [f"target_{prod.replace('ind_', '').replace('_ult1', '')}" for prod in products]
    # Drop rows where any target is null (last observation per customer)
    df_engineered = df_engineered.drop_nulls(subset=target_names)
    logger.info(f'Last month per customer filtered out')
    logger.info(f'Data size after filtering last month per customer: {df_engineered.height:,} rows x {df_engineered.width:,} columns')

    logger.info('DONE: All targets created')

    # Save full preprocessed data with targets
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df_engineered.write_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f'Full preprocessed and engineered data saved to: {data_dir}/{preprocessed_data_file}')
    
    # Save preprocessed data summary
    df_engineered_summary = df_engineered.describe()
    df_engineered_summary.write_parquet(f"{results_dir}/{preprocessed_data_summary_file}")
    logger.info(f'Full preprocessed and engineered data summary saved to: {results_dir}/{preprocessed_data_summary_file}')
    
    del df_engineered_summary
    gc.collect()

    return df_engineered   

def run_target_engineering():
    '''
        Run target engineering pipeline.
    '''
    logger.info('Starting target engineering pipeline')
    df = engineer_targets(DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, PREPROCESSED_DATA_SUMMARY_FILE)
    del df
    gc.collect()
    logger.info('Target engineering pipeline completed successfully')


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_target_engineering()


# ---------- All exports ---------- #
__all__ = ['run_target_engineering', 'engineer_targets']