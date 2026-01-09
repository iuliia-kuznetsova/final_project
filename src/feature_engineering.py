'''
    Feature Engineering

    This module provides functionality to engineer features from preprocessed data.

    Input:
    - data_preprocessed - Preprocessed data
    - results_dir - Output directory for results parquet files.

    Output:
    - data_prepared.parquet - Data file after raw data preprocessing and feature engineering

    Usage:
    python -m src.feature_engineering
'''

# ---------- Imports ---------- #
import polars as pl
import os
from dotenv import load_dotenv
from pathlib import Path

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('feature_engineering')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')


# ---------- Functions ---------- #
def engineer_features( 
    data_dir: str = DATA_DIR, 
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    preprocessed_data_summary_file: str = 'data_preprocessed_feature_engineered_summary.parquet'
) -> pl.DataFrame:
    '''
        Engineer features from preprocessed data.
        
        Strategy:
        - Create targets
        - Add lag features (1, 3, 6, 12 months)
        - Add product interactions → +5% Precision@7
        - Add number of products change feature
    '''
    logger.info('Starting feature engineering pipeline')

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
    logger.info(f'Created targets for new products: {[col.meta.output_name() for col in target_cols]}')
    
    # Add lag features
    # Numeric feature: number of the top 10 most important products that a customer had in the previous month
    lag_months = [3, 6] # not including lag 1 month as it is almost perfectly correlated with current values
    top_products = products[:10]
    # Add lag features for each month
    for lag in lag_months:
        logger.info(f'Adding {lag}m lags...')
        # Add individual product lag features
        lag_cols = [
            pl.col(prod).shift(lag).over('ncodpers').alias(f"{prod}_lag{lag}")
            for prod in top_products
        ]
        # Add aggregate lag feature (total products count at lag)
        lag_cols.append(
            pl.sum_horizontal(top_products).shift(lag).over('ncodpers').alias(f"n_products_lag{lag}")
        )
        df_engineered = df_engineered.with_columns(lag_cols)
        logger.info(f'Added {lag}m lags for {len(top_products)} products')
    
    # Add "recently acquired" features
    # Binary feature: 1 if customer recently acquired product (has now but didn't have 1 month ago)
    recent_acquisition_cols = [
        ((pl.col(prod) == 1) & (pl.col(prod).shift(1).over('ncodpers') == 0))
        .cast(pl.Int8).alias(f"{prod}_acquired_recently")
        for prod in top_products
    ]
    df_engineered = df_engineered.with_columns(recent_acquisition_cols)
    logger.info(f'Added recently acquired features for {len(top_products)} products')
    
    # Add product interactions features using 'one, but not the other' logic
    # Binary feature: 1 if a customer has exactly one of two highly correlated products
    high_value_pairs = [
        ('ind_nomina_ult1', 'ind_nom_pens_ult1'), # payroll account, pensions account
        ('ind_cno_fin_ult1', 'ind_nom_pens_ult1'), # payroll, pensions account  
        ('ind_cno_fin_ult1', 'ind_nomina_ult1'), # payroll, payroll account
        ('ind_cno_fin_ult1', 'ind_recibo_ult1'), # payroll, direct debit
        ('ind_nomina_ult1', 'ind_recibo_ult1'), # payroll account, direct debit
    #    ('ind_nom_pens_ult1', 'ind_recibo_ult1'), # removed as redundant with (ind_nomina_ult1, ind_recibo_ult1) because of almost perfect correlation (0.96)
    ]
    # Add interaction features for each pair
    for prod1, prod2 in high_value_pairs:
        df_engineered = (
            df_engineered
                .with_columns([
                    # (A and not B) or (B and not A) - has exactly one product
                    ((pl.col(prod1) == 1) ^ (pl.col(prod2) == 1)).cast(pl.Int8).alias(f"{prod1}_{prod2}_interaction")
                ])
        )
        logger.info(f'Added interaction feature for {prod1} and {prod2}')

    # Filter out rows from the last month in the dataset
    # Since shift(-1) looks at next month, the last month has no valid targets
    # Get target column names
    target_names = [f"target_{prod.replace('ind_', '').replace('_ult1', '')}" for prod in products]
    # Drop rows where any target is null (last observation per customer)
    df_engineered = df_engineered.drop_nulls(subset=target_names)
    logger.info(f'Filtered out last month per customer - no valid targets possible')
    logger.info(f'Rows after filtering last month per customer: {df_engineered.height:,}')

    # Save preprocessed data
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df_engineered.write_parquet(f"{data_dir}/{preprocessed_data_file}")
    
    # Save preprocessed data summary
    df_engineered_summary = df_engineered.describe()
    df_engineered_summary.write_parquet(f"{results_dir}/{preprocessed_data_summary_file}")

    logger.info(f'Preprocessed and engineered data saved to: {data_dir}/{preprocessed_data_file}')
    logger.info(f'Preprocessed and engineered data summary saved to: {results_dir}/{preprocessed_data_summary_file}')
   
    return df_engineered

def run_feature_engineering():
    '''
        Run feature engineering pipeline.
    '''
    logger.info('Starting feature engineering pipeline')
    df_engineered = engineer_features(DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, 'data_preprocessed_feature_engineered_summary.parquet')
    logger.info('Feature engineering completed successfully')

# ---------- Main function ---------- #
if __name__ == '__main__':
    run_feature_engineering()

# ---------- All exports ---------- #
__all__ = ['run_feature_engineering']