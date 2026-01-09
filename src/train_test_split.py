'''
    Train/Test Split

    This module provides functionality to split feature-engineered data 
    into train and test sets for model training.

    Strategy:
    - Use temporal split (not random) to avoid data leakage
    - Train on earlier months, test on later month(s)
    - Separate features (X) from targets (y)

    Input:
    - data_dir - Directory with preprocessed/engineered data
    - results_dir - Output directory for split data files

    Output:
    - X_train.parquet, y_train.parquet - Training features and targets
    - X_test.parquet, y_test.parquet - Test features and targets

    Usage:
    python -m src.train_test_split
'''

# ---------- Imports ---------- #
import polars as pl
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Tuple

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('train_test_split')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Random state
RANDOM_STATE = int(os.getenv('RANDOM_STATE', 42))

# Directories
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')

# Ids columns
ID_COLS = os.getenv('ID_COLS', ['ncodpers', 'fecha_dato'])


# ---------- Functions ---------- #
def get_feature_and_target_columns(
    df: pl.DataFrame,
    id_cols: List[str] = ID_COLS
) -> Tuple[List[str], List[str]]:
    '''
        Identify feature columns and target columns from the dataframe.
        Excludes datetime columns (CatBoost can't handle them directly).
    '''

    # Target columns start with 'target_'
    target_cols = [col for col in df.columns if col.startswith('target_')]
    
    # Get datetime column names (Date and Datetime types)
    datetime_cols = [
        col for col in df.columns 
        if df[col].dtype in (pl.Date, pl.Datetime, pl.Time)
    ]
    
    # Feature columns are everything except ids, targets, and datetime columns
    feature_cols = [
        col for col in df.columns 
        if col not in id_cols 
        and col not in target_cols
        and col not in datetime_cols
    ]
    
    if datetime_cols:
        logger.info(f'Excluded datetime columns from features: {datetime_cols}')
    
    return feature_cols, target_cols

def temporal_train_test_split(
    data_dir: str = DATA_DIR,
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    test_months: int = 1,
    save_splits: bool = True
) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    '''
        Split data into train and test sets using temporal splitting.
    '''
    logger.info('Starting train/test split pipeline')
    
    # Load preprocessed and feature-engineered data
    df = pl.read_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f'Loaded data from: {data_dir}/{preprocessed_data_file}')
    logger.info(f'Total rows: {df.height:,}, Total columns: {df.width}')
    
    # Get unique dates and sort them
    dates = df.select('fecha_dato').unique().sort('fecha_dato')
    unique_dates = dates['fecha_dato'].to_list()
    logger.info(f'Date range: {unique_dates[0]} to {unique_dates[-1]} ({len(unique_dates)} months)')
    
    # Define train/test split date
    # Test set: last `test_months` month(s)
    # Train set: all earlier months
    test_start_idx = len(unique_dates) - test_months
    train_dates = unique_dates[:test_start_idx]
    test_dates = unique_dates[test_start_idx:]
    logger.info(f'Train dates: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} months)')
    logger.info(f'Test dates: {test_dates[0]} to {test_dates[-1]} ({len(test_dates)} months)')
    
    # Split data by date
    df_train = df.filter(pl.col('fecha_dato').is_in(train_dates))
    df_test = df.filter(pl.col('fecha_dato').is_in(test_dates))
    logger.info(f'Train rows: {df_train.height:,}')
    logger.info(f'Test rows: {df_test.height:,}')
    
    # Get feature and target columns
    feature_cols, target_cols = get_feature_and_target_columns(df)
    logger.info(f'Number of features: {len(feature_cols)}')
    logger.info(f'Number of targets: {len(target_cols)}')
       
    # Separate features (X) and targets (y)
    X_train = df_train.select(feature_cols)
    X_test = df_test.select(feature_cols)
    y_train = df_train.select(target_cols)
    y_test = df_test.select(target_cols)
    logger.info('DONE: Train/test split completed')

    # Save splits if requested
    if save_splits:
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        X_train.write_parquet(f"{data_dir}/X_train.parquet")
        X_test.write_parquet(f"{data_dir}/X_test.parquet")
        y_train.write_parquet(f"{data_dir}/y_train.parquet")
        y_test.write_parquet(f"{data_dir}/y_test.parquet")
        logger.info(f'DONE: Train/test splits saved to: {data_dir}/')
    
    return X_train, X_test, y_train, y_test


def load_splits(data_dir: str = DATA_DIR) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    '''
        Load previously saved train/test splits.
    '''
    X_train = pl.read_parquet(f"{data_dir}/X_train.parquet")
    X_test = pl.read_parquet(f"{data_dir}/X_test.parquet")
    y_train = pl.read_parquet(f"{data_dir}/y_train.parquet")
    y_test = pl.read_parquet(f"{data_dir}/y_test.parquet")
    
    logger.info(f'Loaded train/test splits from: {data_dir}/')
    logger.info(f'   X_train: {X_train.height:,} rows, {X_train.width} columns')
    logger.info(f'   X_test: {X_test.height:,} rows, {X_test.width} columns')
    logger.info(f'   y_train: {y_train.height:,} rows, {y_train.width} columns')
    logger.info(f'   y_test: {y_test.height:,} rows, {y_test.width} columns')
    
    return X_train, X_test, y_train, y_test

def run_train_test_split():
    '''
    Run train/test split pipeline.
    '''
    logger.info('Starting train/test split pipeline')
    X_train, X_test, y_train, y_test = temporal_train_test_split(
        DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, 
        test_months=1, 
        save_splits=True
    )
    logger.info('Train/test split completed successfully')

# ---------- Main function ---------- #
if __name__ == '__main__':
    run_train_test_split()

# ---------- All exports ---------- #
__all__ = ['run_train_test_split']