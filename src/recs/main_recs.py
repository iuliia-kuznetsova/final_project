'''
    CLI entry point for bank products recommendation system.

    Usage examples:
        python -m src.recs.main_recs                   # run full pipeline
        python -m src.recs.main_recs --skip-download   # skip raw data download if already present
    
    Memory optimization:
        - Explicit gc.collect() calls after each pipeline step to free intermediate data
        - Training data cleaned up after modelling completes
'''

# ---------- Imports ---------- #
import os
import sys
import gc
import argparse
from pythonjsonlogger import jsonlogger
import traceback
from dotenv import load_dotenv
import polars as pl

from src.logging_setup import setup_logging
from src.recs.data_loading import run_data_loading
from src.recs.data_preprocessing import run_preprocessing
from src.recs.feature_engineering import run_feature_engineering
from src.recs.target_engineering import run_target_engineering
from src.recs.train_test_split import run_train_test_split
from src.recs.modelling_ovr import run_modelling_ovr, OvRGroupModel, load_training_data, get_product_names


# ---------- Memory Helper ---------- #
def log_memory_cleanup(logger, step_name: str):
    '''Force garbage collection and log memory cleanup after a pipeline step.'''
    collected = gc.collect()
    logger.info(f'Memory cleanup after {step_name}: {collected} objects collected')


# ---------- Logging setup ---------- #
logger = setup_logging('main_recs')


# ---------- Argument Parser ---------- #
def parse_args():
    '''Parse command line arguments.'''
    parser = argparse.ArgumentParser(
        description='Bank Products Recommendation System Pipeline'
    )
    parser.add_argument(
        '--skip-download',
        action='store_true',
        help='Skip raw data download if already present'
    )
    parser.add_argument(
        '--skip-modelling',
        action='store_true',
        help='Skip model training'
    )
    return parser.parse_args()


# ---------- Main pipeline---------- #
def main(args):
    '''
        Main entry point: 
        1. Load environment variables
        2. Download raw data (if needed)
        3. Preprocess data
        4. Feature engineering
        5. Target engineering
        6. Sample data and split into train/test sets
        7. Train models: OvR CatBoost (Optuna optimization + CV on sampled data)
        8. Evaluate models: OvR CatBoost
        9. Generate recommendations
    '''
    
    # ---------- Step 1: Load environment variables ---------- #
    print('\n' + '='*80)
    logger.info('STEP 1: Loading environment variables')
    print('='*80)
    
    load_dotenv()

    DATA_DIR = os.getenv('DATA_DIR', './data')
    MODELS_DIR = os.getenv('MODELS_DIR', './models')
    RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
    RAW_DATA_FILE = os.getenv('RAW_DATA_FILE', 'train_ver2.csv')
    PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
    
    logger.info('DONE: Loading environment variables completed successfully')
    logger.info(f'INFO: Data directory: {DATA_DIR}')
    logger.info(f'INFO: Models directory: {MODELS_DIR}')
    logger.info(f'INFO: Results directory: {RESULTS_DIR}')
    logger.info(f'STEP 1 DONE')
    
    # ---------- Step 2: Download raw data (if not skipped) ---------- #
    if not args.skip_download:
        print('\n' + '='*80)
        logger.info('STEP 2: Downloading raw data')
        print('='*80)
        
        try:
            run_data_loading()
            log_memory_cleanup(logger, 'data_loading')
            logger.info('STEP 2 DONE')
        except Exception as e:
            logger.error(f'ERROR: Failed to download raw data: {e}')
            sys.exit(1)
    else:
        print('\n' + '='*80)
        logger.info('STEP 2: Skipping raw data download (--skip-download flag)')
        print('='*80)
        
        raw_data_path = os.path.join(DATA_DIR, RAW_DATA_FILE)
        if not os.path.exists(raw_data_path):
            logger.error(f'ERROR: Missing raw data file: {raw_data_path}')
            logger.info(f'INFO: Run without --skip-download to download raw data')
            sys.exit(1)
        
        logger.info('STEP 2 DONE')
    
    # ---------- Step 3: Run preprocessing pipeline ---------- #
    print('\n' + '='*80)
    logger.info('STEP 3: Preprocessing data')
    print('='*80)
    
    try:
        run_preprocessing()
        
        # Verify that the required preprocessed data file exists
        if PREPROCESSED_DATA_FILE not in os.listdir(DATA_DIR):
            logger.error(f'ERROR: Missing preprocessed data file: {DATA_DIR}/{PREPROCESSED_DATA_FILE}')
            sys.exit(1)
        
        log_memory_cleanup(logger, 'preprocessing')
        logger.info('STEP 3 DONE')
        
    except Exception as e:
        logger.error(f'ERROR: Preprocessing failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Step 4: Feature Engineering ---------- #
    print('\n' + '='*80)
    logger.info('STEP 4: Feature Engineering')
    print('='*80)
    
    try:
        run_feature_engineering()
        log_memory_cleanup(logger, 'feature_engineering')
        logger.info('STEP 4 DONE')
        
    except Exception as e:
        logger.error(f'ERROR: Feature engineering failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Step 5: Target Engineering ---------- #
    print('\n' + '='*80)
    logger.info('STEP 5: Target Engineering')
    print('='*80)
    
    try:
        run_target_engineering()
        log_memory_cleanup(logger, 'target_engineering')
        logger.info('STEP 5 DONE')
        
    except Exception as e:
        logger.error(f'ERROR: Target engineering failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    # ---------- Step 6: Split data into train/test sets ---------- #
    print('\n' + '='*80)
    logger.info('STEP 6: Splitting data into train/test sets')
    print('='*80)
    
    try:
        run_train_test_split()

        # Verify that all needed train/test split files exist
        train_test_split_files = ['X_train.parquet', 'X_test.parquet', 'y_train.parquet', 'y_test.parquet']
        missing_train_test_split_files = [f for f in train_test_split_files if not os.path.exists(os.path.join(DATA_DIR, f))]
        
        if missing_train_test_split_files:
            logger.error(f'ERROR: Missing train/test split files: {missing_train_test_split_files}')
            logger.info('INFO: Check train/test split pipeline to generate train/test split data')
            sys.exit(1)
        
        log_memory_cleanup(logger, 'train_test_split')
        logger.info('STEP 6 DONE')

    except Exception as e:
        logger.error(f'ERROR: Splitting data into train/test sets failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # Skip modelling if flag is set
    if args.skip_modelling:
        print('\n' + '='*80)
        logger.info('STEP 7-9: Skipping modelling (--skip-modelling flag)')
        print('='*80)
        logger.info('Pipeline completed (modelling skipped)')
        return

    # ---------- Step 7: Modelling ---------- #     
    print('\n' + '='*80)
    logger.info('STEP 7: Modelling OvR Grouped CatBoost')
    print('='*80)
    
    try:
        # Load training data
        X_train, X_test, y_train, y_test = load_training_data(DATA_DIR)
        target_names = get_product_names(y_train)
        
        # Detect categorical features
        cat_features = [
            col for col in X_train.columns
            if X_train[col].dtype == 'category' or X_train[col].dtype == 'object'
        ]
        
        # Initialize and train model
        model = OvRGroupModel(
            cat_features=cat_features,
            models_dir=MODELS_DIR,
            results_dir=RESULTS_DIR,
            data_dir=DATA_DIR
        )
        model.fit(
            X_train, y_train,
            X_val=X_test, y_val=y_test,
            target_names=target_names
        )
        
        # Free training data memory after model fitting (keep X_test, y_test for evaluation)
        del X_train, y_train
        log_memory_cleanup(logger, 'model_training')
        logger.info('STEP 7 DONE')

    except Exception as e:
        logger.error(f'ERROR: Modelling OvR Grouped CatBoost failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 8: Evaluation ---------- #
    print('\n' + '='*80)
    logger.info('STEP 8: Evaluating OvR Grouped CatBoost')
    print('='*80)
    
    try:
        results = model.evaluate(X_test, y_test)
        logger.info(f"Mean AUC: {results['overall']['mean_auc']:.4f}")
        log_memory_cleanup(logger, 'evaluation')
        logger.info('STEP 8 DONE')
    except Exception as e:
        logger.error(f'ERROR: Evaluating OvR Grouped CatBoost failed: {e}')
        traceback.print_exc()
        sys.exit(1)

    # ---------- Step 9: Recommendations & Save ---------- #
    print('\n' + '='*80)
    logger.info('STEP 9: Generating recommendations and saving model')
    print('='*80)
    
    try:
        # Generate sample recommendations (use small subset to save memory)
        X_test_sample = X_test.head(100)
        recommendations = model.recommend(X_test_sample)
        logger.info(f'Generated recommendations for {len(recommendations)} customers')
        
        # Save model
        model_path = model.save('ovr_grouped_santander')
        logger.info(f'Model saved to: {model_path}')
        
        # Final cleanup: free test data and model from memory
        del X_test, y_test, X_test_sample, recommendations, model
        log_memory_cleanup(logger, 'final_cleanup')
        
        logger.info('STEP 9 DONE')
    except Exception as e:
        logger.error(f'ERROR: Generating recommendations failed: {e}')
        traceback.print_exc()
        sys.exit(1)
    
    logger.info('Pipeline completed successfully!')


# ---------- Main function ---------- #
if __name__ == '__main__':
    args = parse_args()
    main(args)




