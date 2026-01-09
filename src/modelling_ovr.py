'''
    One-vs-Rest (OvR) Group-based Modelling with CatBoost

    This module provides OvR classification with:
    - Products grouped by prevalence: frequent, mid, rare
    - Separate OvR CatBoost model per group
    - TimeSeriesSplit 5-fold cross-validation
    - Optuna hyperparameter optimization per group
    - Per-product threshold optimization
    - MLflow experiment tracking
    - Production: group → model mapping
    - Inference: top-7 recommendations

    Strategy:
    - Group 24 products by prevalence into 3 groups
    - Train separate OvR(CatBoost) for each group with optimized hyperparameters
    - Optimize thresholds per product on validation set
    - For inference: combine predictions from all groups, rank top-7

    Input:
    - X_train, y_train - Training features and targets
    - X_test, y_test - Test features and targets

    Output:
    - 3 trained OvR models (frequent, mid, rare groups)
    - Evaluation metrics logged to MLflow
    - Top-7 product recommendations

    Usage:
    python -m src.modelling_ovr
'''

# ---------- Imports ---------- #
import polars as pl
import numpy as np
import pandas as pd
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Union
from datetime import datetime

from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    log_loss
)
from dotenv import load_dotenv
from tqdm import tqdm
import optuna
from optuna.samplers import TPESampler

try:
    import mlflow
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('modelling_ovr')


# ---------- Config ---------- #
load_dotenv()
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
# Models directory
MODELS_DIR = os.getenv('MODELS_DIR', './models')

# MLflow experiment name
MLFLOW_EXPERIMENT = os.getenv('MLFLOW_EXPERIMENT', 'ovr_grouped_catboost')

# Number of CV folds
N_SPLITS = int(os.getenv('N_SPLITS', 5))

# Top-K recommendations
TOP_K = int(os.getenv('TOP_K', 7))

# Prevalence group thresholds (percentages)
FREQUENT_THRESHOLD = float(os.getenv('FREQUENT_THRESHOLD', 5.0))
RARE_THRESHOLD = float(os.getenv('RARE_THRESHOLD', 1.0))

# Optimization parameters
OPTIMIZE = bool(os.getenv('OPTIMIZE', True)) # Run Optuna hyperparameter optimization
N_TRIALS = int(os.getenv('N_TRIALS', 15)) # Optuna trials per group
OPTUNA_TIMEOUT = int(os.getenv('OPTUNA_TIMEOUT', 180)) # Optuna timeout per group optimization (seconds)
RUN_CV = bool(os.getenv('RUN_CV', True)) # Run 5-fold TimeSeriesSplit CV

# ---------- Helper Functions ---------- #
def load_training_data(
    data_dir: str = DATA_DIR
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
        Load train/test as pandas DataFrames.
        Uses Polars to read parquet (handles Categorical types) then converts to pandas.
    '''

    # Load with Polars first to handle Categorical columns properly
    X_train = pl.read_parquet(f"{data_dir}/X_train.parquet").to_pandas()
    X_test = pl.read_parquet(f"{data_dir}/X_test.parquet").to_pandas()
    y_train = pl.read_parquet(f"{data_dir}/y_train.parquet").to_pandas()
    y_test = pl.read_parquet(f"{data_dir}/y_test.parquet").to_pandas()
    
    logger.info(f'Loaded train data: X_train={X_train.shape}, y_train={y_train.shape}')
    logger.info(f'Loaded test data: X_test={X_test.shape}, y_test={y_test.shape}')

    return X_train, X_test, y_train, y_test

def get_product_names(
    y_train: pd.DataFrame
) -> List[str]:
    '''
        Extract product names from target columns.
    '''
    return [col for col in y_train.columns if col.startswith('target_')]


# ---------- OvR Group Model Class ---------- #
class OvRGroupModel:
    '''
        One-vs-Rest model with products grouped by prevalence.
        
        Features:
        - 3 groups: frequent (>5%), mid (1-5%), rare (<1%)
        - Separate OvR(CatBoost) per group
        - Optuna hyperparameter optimization per group
        - TimeSeriesSplit 5-fold CV
        - Per-product threshold optimization
        - MLflow tracking
        - Production: group → model mapping
        - Inference: top-7 recommendations
    '''
    
    def __init__(
        self,
        cat_features: Optional[List[str]] = None,
        random_state: int = RANDOM_STATE,
        n_splits: int = N_SPLITS,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        mlflow_experiment: Optional[str] = MLFLOW_EXPERIMENT,
        frequent_threshold: float = FREQUENT_THRESHOLD,
        rare_threshold: float = RARE_THRESHOLD,
        optimize: bool = OPTIMIZE,
        n_trials: int = N_TRIALS,
        optuna_timeout: int = OPTUNA_TIMEOUT,
        run_cv: bool = RUN_CV,
        top_k: int = TOP_K
    ):
        '''
            Initialize OvR Group model:
                - Separate OvR model per group
                - Grouped products by prevalence: 
                frequent (>FREQUENT_THRESHOLD%), mid (RARE_THRESHOLD-FREQUENT_THRESHOLD%), rare (<RARE_THRESHOLD%)
                - CatBoost base estimator for each group
                - Optuna hyperparameter optimization for each group
                - TimeSeriesSplit N_SPLITS-fold CV for each group
                - Per-product threshold optimization
                - MLflow tracking for each group
                - Production: group model mapping
                - Inference: top-K recommendations for each group
        '''
        self.cat_features = cat_features or []
        self.random_state = random_state
        self.n_splits = n_splits
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.mlflow_experiment = mlflow_experiment
        self.frequent_threshold = frequent_threshold
        self.rare_threshold = rare_threshold
        self.optimize = optimize
        self.n_trials = n_trials
        self.optuna_timeout = optuna_timeout
        self.run_cv = run_cv
        self.top_k = top_k

        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # OvR models per group: {'frequent': OvR, 'mid': OvR, 'rare': OvR}
        self.models: Dict[str, OneVsRestClassifier] = {}
        
        # Products in each group: {'frequent': [products], 'mid': [...], 'rare': [...]}
        self.product_groups: Dict[str, List[str]] = {}
        
        # Best hyperparameters per group
        self.best_params: Dict[str, Dict[str, Any]] = {}
        
        # Thresholds per product (across all groups)
        self.thresholds: Dict[str, float] = {}
        
        # CV scores per group
        self.cv_scores: Dict[str, List[float]] = {}
        
        # All products (ordered)
        self.all_products: List[str] = []
        
        # Feature metadata
        self.feature_names: List[str] = []
        self.cat_feature_indices: List[int] = []
        
        # Setup MLflow
        if MLFLOW_AVAILABLE and mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)
            logger.info(f'MLflow experiment: {mlflow_experiment}')
        
        logger.info('OvRGroupModel initialized')
    
    # ----------- Data Preparation ------------ #
    def _prepare_data(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        y: Optional[Union[pl.DataFrame, pd.DataFrame]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        '''
            Convert data to pandas DataFrames.
            Handles NaN values in categorical columns (CatBoost requirement).
        '''

        # Convert X
        if isinstance(X, pl.DataFrame):
            X_pd = X.to_pandas()
        else:
            X_pd = X.copy()
        
        # Convert categorical columns to string for CatBoost compatibility
        # CatBoost cannot handle None/NaN in categorical features
        for col in X_pd.columns:
            if X_pd[col].dtype.name == 'category' or X_pd[col].dtype == 'object':
                # Convert to string first (NaN becomes 'nan'), then replace with 'missing'
                X_pd[col] = X_pd[col].astype(str).replace({'nan': 'missing', 'None': 'missing', '<NA>': 'missing'})
        
        # Convert y
        if y is None:
            y_pd = None
        elif isinstance(y, pl.DataFrame):
            y_pd = y.to_pandas()
        else:
            y_pd = y.copy()
        
        y_shape = y_pd.shape if y_pd is not None else None
        logger.info(f'Prepared data: X_pd={X_pd.shape}, y_pd={y_shape}')

        return X_pd, y_pd
    
    def _get_cat_feature_indices(
        self, X: pd.DataFrame
    ) -> List[int]:
        '''
            Get indices of categorical features.
        '''

        if not self.cat_features:
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if X[col].dtype == 'object' or X[col].dtype.name == 'category'
            ]
        else:
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if col in self.cat_features
            ]

        logger.info(f'Categorical features: {self.cat_features}')
        logger.info(f'Categorical feature indices: {cat_indices}')

        return cat_indices
    
    def _group_products_by_prevalence(
        self, 
        y: pd.DataFrame
    ) -> Dict[str, List[str]]:
        '''
            Group products into frequent, mid, and rare based on prevalence.
        '''

        # Calculate prevalence as a percentage of total customers 
        # who don't have product current month, but will newly acquire it next month
        prevalence = y.fillna(0).mean() * 100
        
        # Sort products by prevalence descending
        sorted_prevalence = prevalence.sort_values(ascending=False)

        frequent_products = []
        mid_products = []
        rare_products = []

        for product in sorted_prevalence.index:
            if sorted_prevalence[product] > self.frequent_threshold:
                frequent_products.append(product)
            elif sorted_prevalence[product] < self.rare_threshold:
                rare_products.append(product)
            else:
                mid_products.append(product)
        
        groups = {
            'frequent': frequent_products,
            'mid': mid_products,
            'rare': rare_products
        }
        ## Calculate percentile-based cutoffs (top third = frequent, bottom third = rare)
        #frequent_cutoff = int(n_products / 3)  # Top ~8 products
        #rare_cutoff = int(2 * n_products / 3)  # Bottom ~8 products
        #
        ## Get product names by position in sorted order
        #frequent_products = sorted_prevalence.index[:frequent_cutoff].tolist()
        #mid_products = sorted_prevalence.index[frequent_cutoff:rare_cutoff].tolist()
        #rare_products = sorted_prevalence.index[rare_cutoff:].tolist()
       #
       #groups = {
       #    'frequent': frequent_products,
       #    'mid': mid_products,
       #    'rare': rare_products
       #}
       #
       ## Calculate the actual threshold values for logging
       #if frequent_products:
       #    freq_threshold = prevalence[frequent_products[-1]]  # Lowest in frequent group
       #else:
       #    freq_threshold = float('inf')
       #if mid_products:
       #    rare_threshold = prevalence[mid_products[-1]]  # Lowest in mid group
       #else:
       #    rare_threshold = 0
       #
        logger.info('Created product groups by prevalence (percentile-based):')
        logger.info(f"  Frequent (prevalence >={self.frequent_threshold:.4f}%): {len(groups['frequent'])} products")
        logger.info(f"  Mid (prevalence {self.rare_threshold:.4f}%-{self.frequent_threshold:.4f}%): {len(groups['mid'])} products")
        logger.info(f"  Rare (prevalence <{self.rare_threshold:.4f}%): {len(groups['rare'])} products")
        
        for group_name, products in groups.items():
            if products:
                prevalences = [prevalence[p] for p in products]
                logger.info(f"    {group_name}: computed prevalence range [{min(prevalences):.4f}%, {max(prevalences):.4f}%]")
        
        return groups
      
    # ----------- Hyperparameter Optimization ------------ #
    def _create_optuna_objective(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        cat_feature_indices: List[int]
    ):
        '''
            Create Optuna objective with TimeSeriesSplit CV for OvR.
        '''

        def objective(trial: optuna.Trial) -> float:

            # Hyperparameter search space (reduced for speed)
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500, step=100),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                # Fixed params
                'loss_function': 'Logloss',
                'random_seed': self.random_state,
                'verbose': False,
                'thread_count': -1,
                'auto_class_weights': 'Balanced'
            }
            
            # TimeSeriesSplit CV
            tscv = TimeSeriesSplit(n_splits=self.n_splits)
            auc_scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y[train_idx], y[val_idx]
                
                # Create OvR with CatBoost base estimator
                base_estimator = CatBoostClassifier(
                    cat_features=cat_feature_indices if cat_feature_indices else None,
                    **params
                )
                ovr = OneVsRestClassifier(base_estimator, n_jobs=1)
                
                # Train
                ovr.fit(X_tr, y_tr)
                
                # Predict probabilities
                y_proba = ovr.predict_proba(X_val)
                
                # Calculate mean AUC across products
                fold_aucs = []
                for i in range(y_val.shape[1]):  # Loop over products
                    positives = y_val[:, i].sum()  # Number of 1s in this product column
                    
                    # Check whether the case is valid for AUC calculation
                    if positives > 0 and positives < len(y_val):
                        # Valid: 1 ≤ positives ≤ n_samples-1
                        auc = roc_auc_score(y_val[:, i], y_proba[:, i])
                        fold_aucs.append(auc)

                # Add fold score only if any products were valid
                if fold_aucs:  # len(fold_aucs) > 0
                    auc_scores.append(np.mean(fold_aucs))  # Mean AUC across valid products
            
            return np.mean(auc_scores) if auc_scores else 0.0
        
        return objective
    
    def _optimize_group_hyperparameters(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        group_name: str
    ) -> Dict[str, Any]:
        '''
            Run Optuna hyperparameter optimization for a group.
        '''
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Create Optuna objective
        objective = self._create_optuna_objective(
            X, y, self.cat_feature_indices
        )
        
        # Don't show Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Optimize with timeout for faster execution
        study.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.optuna_timeout,
            show_progress_bar=True
        )
        
        # Build full params
        best_params = study.best_params.copy()
        best_params.update({
            'loss_function': 'Logloss',
            'random_seed': self.random_state,
            'verbose': False,
            'thread_count': -1,
            'auto_class_weights': 'Balanced'
        })
        
        logger.info(f'DONE: Hyperparameter optimization for {group_name} group completed')
        logger.info(f'   Best CV AUC of {group_name} group: {study.best_value:.4f}')
        
        return best_params

    # ----------- CV Evaluation ------------ #
    def _cv_evaluate_group(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        params: Dict[str, Any]
    ) -> List[float]:
        '''
            Perform N_SPLITS-fold TimeSeriesSplit CV for a group.
        '''

        # TimeSeriesSplit CV
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_aucs = []
        
        # Loop over folds
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Create OvR with CatBoost base estimator
            base_estimator = CatBoostClassifier(
                cat_features=self.cat_feature_indices if self.cat_feature_indices else None,
                **params
            )
            ovr = OneVsRestClassifier(base_estimator, n_jobs=1)
            
            # Train
            ovr.fit(X_tr, y_tr)
            
            # Predict probabilities
            y_proba = ovr.predict_proba(X_val)
            
            # Calculate AUC for each product
            product_aucs = []
            for i in range(y_val.shape[1]): # Loop over products
                positives = y_val[:, i].sum() # Number of 1s in this product column
                
                # Check whether the case is valid for AUC calculation
                if positives > 0 and positives < len(y_val): # Valid: 1 ≤ positives ≤ n_samples-1
                    auc = roc_auc_score(y_val[:, i], y_proba[:, i])
                    product_aucs.append(auc)
            
            if product_aucs: # Add fold score only if any products were valid
                fold_aucs.append(np.mean(product_aucs)) # Mean AUC across valid products
        
        logger.info(f'DONE: CV evaluation for group completed')
        logger.info(f'   Fold AUCs: {fold_aucs}')

        return fold_aucs
    
    # ----------- Threshold Optimization ------------ #
    def _optimize_thresholds_for_group(
        self,
        products: List[str],
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        '''
            Optimize classification thresholds per product using F1 score.
        '''
        thresholds = {}

        # Optimize thresholds for each product
        for i, product in enumerate(products):
            y_true_i = y_true[:, i]  # True labels for this product
            y_proba_i = y_proba[:, i]  # Predicted probabilities for this product

            # If there are no positive cases, set threshold to 0.5
            if y_true_i.sum() == 0:
                thresholds[product] = 0.5
                continue
            
            # Calculate precision, recall, and threshold
            precision, recall, thresh = precision_recall_curve(y_true_i, y_proba_i)

            # Calculate F1 scores
            # F1 score is the harmonic mean of precision and recall
            f1_scores = np.where(
                (precision[:-1] + recall[:-1]) > 0,  # Check if division by zero
                2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1]),  # F1 score formula
                0  # If no positive cases, set F1 score to 0
            )
            
            # Find the best threshold that maximizes F1 score
            if len(f1_scores) > 0 and np.max(f1_scores) > 0:  # Check if there are valid thresholds
                best_idx = np.argmax(f1_scores)  # Index of the best threshold
                thresholds[product] = float(thresh[best_idx])  # Set the threshold for this product
            else:
                thresholds[product] = 0.5  # If no valid thresholds, set threshold to 0.5
            
        logger.info(f'DONE: Thresholds optimization for {len(products)} products completed')
        logger.info(f'   Product thresholds: {thresholds}')

        return thresholds
    
    # ----------- Training ------------ #
    def fit(
        self,
        X_train: Union[pl.DataFrame, pd.DataFrame],
        y_train: Union[pl.DataFrame, pd.DataFrame],
        X_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        target_names: Optional[List[str]] = None
    ) -> 'OvRGroupModel':
        '''
            Train OvR models for all product groups.
        '''

        # Prepare data
        X_pd, y_pd = self._prepare_data(X_train, y_train)
        self.feature_names = list(X_pd.columns)
        self.cat_feature_indices = self._get_cat_feature_indices(X_pd)
        
        # Get target names
        if target_names is not None:
            self.all_products = target_names
        elif y_pd is not None:
            self.all_products = list(y_pd.columns)
        
        logger.info(f'Total products: {len(self.all_products)}')
        logger.info(f'Total features: {len(self.feature_names)}')
        logger.info(f'Categorical features: {len(self.cat_feature_indices)}')
        
        # Group products by prevalence
        self.product_groups = self._group_products_by_prevalence(y_pd)
        
        # Prepare validation data
        X_val_pd, y_val_pd = None, None
        if X_val is not None and y_val is not None:
            X_val_pd, y_val_pd = self._prepare_data(X_val, y_val)
        
        # Start MLflow run
        mlflow_run = None
        if MLFLOW_AVAILABLE and self.mlflow_experiment:
            mlflow_run = mlflow.start_run(
                run_name=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            )
            mlflow.log_param('n_products', len(self.all_products))
            mlflow.log_param('n_features', len(self.feature_names))
            mlflow.log_param('n_train_samples', len(X_pd))
            mlflow.log_param('n_frequent', len(self.product_groups['frequent']))
            mlflow.log_param('n_mid', len(self.product_groups['mid']))
            mlflow.log_param('n_rare', len(self.product_groups['rare']))
            mlflow.log_param('optimize', self.optimize)
            mlflow.log_param('n_trials', self.n_trials)
            mlflow.log_param('optuna_timeout', self.optuna_timeout)
            mlflow.log_param('run_cv', self.run_cv)
        
        # Train model for each group with progress bar
        group_names = ['frequent', 'mid', 'rare']
        group_progress = tqdm(group_names, desc='Training Groups', unit='group')
        
        all_cv_aucs = []
        
        try:
            for group_name in group_progress:
                products = self.product_groups[group_name]
                
                if not products:
                    logger.info(f'Skipping {group_name} group: no products')
                    continue
                
                group_progress.set_postfix({'group': group_name, 'products': len(products)})
                logger.info(f'Training {group_name.upper()} group ({len(products)} products)')
                
                # Get target columns for this group (fill nulls with 0, convert to int)
                y_group = y_pd[products].astype(int).values
                
                # Start nested MLflow run for this group
                group_mlflow_run = None
                if MLFLOW_AVAILABLE and mlflow_run:
                    group_mlflow_run = mlflow.start_run(
                        run_name=f'group_{group_name}',
                        nested=True
                    )
                    mlflow.log_param('group_name', group_name)
                    mlflow.log_param('n_products', len(products))
                    mlflow.log_param('products', products[:10])  # Log first 10
                
                try:
                    # Optuna optimization
                    if self.optimize:
                        best_params = self._optimize_group_hyperparameters(
                            X_pd, y_group, group_name
                        )
                        self.best_params[group_name] = best_params
                        
                        if MLFLOW_AVAILABLE and group_mlflow_run:
                            for k, v in best_params.items():
                                if k not in ['verbose', 'thread_count']:
                                    mlflow.log_param(f'hp_{k}', v)
                    else:
                        best_params = {
                            'iterations': 300,
                            'depth': 6,
                            'learning_rate': 0.1,
                            'loss_function': 'Logloss',
                            'random_seed': self.random_state,
                            'verbose': False,
                            'thread_count': -1,
                            'auto_class_weights': 'Balanced'
                        }
                        self.best_params[group_name] = best_params
                    
                    # 5-fold TimeSeriesSplit CV evaluation
                    if self.run_cv:
                        cv_scores = self._cv_evaluate_group(X_pd, y_group, best_params)
                        self.cv_scores[group_name] = cv_scores
                        mean_cv = np.mean(cv_scores) if cv_scores else 0.0
                        std_cv = np.std(cv_scores) if cv_scores else 0.0
                        all_cv_aucs.extend(cv_scores)
                        
                        logger.info(f'{group_name} CV AUC: {mean_cv:.4f} ± {std_cv:.4f}')
                        
                        if MLFLOW_AVAILABLE and group_mlflow_run:
                            mlflow.log_metric('cv_mean_auc', mean_cv)
                            mlflow.log_metric('cv_std_auc', std_cv)
                            for fold_idx, fold_auc in enumerate(cv_scores):
                                mlflow.log_metric(f'cv_fold_{fold_idx}_auc', fold_auc)
                    
                    # Train final model on full training data
                    logger.info(f'Training final {group_name} model')
                    base_estimator = CatBoostClassifier(
                        cat_features=self.cat_feature_indices if self.cat_feature_indices else None,
                        **best_params
                    )
                    model = OneVsRestClassifier(base_estimator, n_jobs=-1)
                    model.fit(X_pd, y_group)
                    
                    # Store model in registry
                    self.models[group_name] = model
                    
                    # Optimize thresholds on validation set
                    if X_val_pd is not None and y_val_pd is not None:
                        y_val_group = y_val_pd[products].astype(int).values
                        y_proba_val = model.predict_proba(X_val_pd)
                        group_thresholds = self._optimize_thresholds_for_group(
                            products, y_val_group, y_proba_val
                        )
                        self.thresholds.update(group_thresholds)
                        logger.info(f'{group_name} thresholds optimized')
                    else:
                        # Default thresholds
                        for product in products:
                            self.thresholds[product] = 0.5
                    
                    # Log model to MLflow
                    if MLFLOW_AVAILABLE and group_mlflow_run:
                        for i, (product, estimator) in enumerate(zip(products, model.estimators_)):
                            if i < 5:  # Log first 5 estimators per group
                                mlflow.catboost.log_model(estimator, f'model_{product}')
                
                finally:
                    if MLFLOW_AVAILABLE and group_mlflow_run:
                        mlflow.end_run()
            
            # Log overall metrics
            if MLFLOW_AVAILABLE and mlflow_run and all_cv_aucs:
                mlflow.log_metric('overall_mean_cv_auc', np.mean(all_cv_aucs))
                mlflow.log_metric('overall_std_cv_auc', np.std(all_cv_aucs))
                mlflow.log_metric('n_groups_trained', len(self.models))
        
        finally:
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
        
        # Save thresholds
        thresholds_file = self.results_dir / 'ovr_group_thresholds.json'
        with open(thresholds_file, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
        logger.info(f'Thresholds saved to: {thresholds_file}')
        
        logger.info(f'DONE: Training completed: {len(self.models)} group models')
        if all_cv_aucs:
            logger.info(f'   Overall CV AUC of all groups: {np.mean(all_cv_aucs):.4f} ± {np.std(all_cv_aucs):.4f}')
        
        return self
    
    # ----------- Inference ------------ #
    def predict_proba(
        self,
        X: Union[pl.DataFrame, pd.DataFrame]
    ) -> pd.DataFrame:
        '''
            Get probability predictions for all products.
        '''

        # Check if models are trained
        if not self.models:
            raise ValueError('No models trained.')
        
        # Prepare data
        X_pd, _ = self._prepare_data(X)
        
        # Collect predictions from all groups
        all_probas = {}
        
        for group_name, model in self.models.items(): # Loop over groups
            products = self.product_groups[group_name]
            y_proba = model.predict_proba(X_pd) # Predict probabilities
            
            for i, product in enumerate(products): # Loop over products
                all_probas[product] = y_proba[:, i] 
        
        # Order by original product order
        ordered_probas = {p: all_probas[p] for p in self.all_products if p in all_probas} # Filter products that are in the model
        
        logger.info(f'DONE: Predictions for all products completed')
        logger.info(f'   Ordered probabilities for {len(ordered_probas)} products: {ordered_probas}')
        
        return pd.DataFrame(ordered_probas)
    
    # ----------- Top-K Recommendations ------------ #
    def predict_top_k(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        apply_thresholds: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        '''
            Get top-K product recommendations per customer.
            Combines all group predictions, ranks by probability.
        '''

        # Get probabilities for all products
        proba_df = self.predict_proba(X)
        # Convert to matrix
        proba_matrix = proba_df.values
        # Get product names
        product_names = list(proba_df.columns)
        # Get number of customers
        n_customers = proba_matrix.shape[0]
        
        # Rank products by probability
        # Get indices of top-k products for each customer
        top_k_indices = np.argsort(-proba_matrix, axis=1)[:, :self.top_k]
        # Get corresponding probabilities for the top-k products
        top_k_probas = np.take_along_axis(proba_matrix, top_k_indices, axis=1)
        
        # Apply threshold filtering
        if apply_thresholds:
            for i in range(n_customers): # Loop over customers
                for j in range(self.top_k): # Loop over top-k products
                    product_idx = top_k_indices[i, j]
                    product_name = product_names[product_idx]
                    threshold = self.thresholds.get(product_name, 0.5)
                    if top_k_probas[i, j] < threshold: # If probability is less than threshold, set to 0
                        top_k_probas[i, j] = 0.0
        
        # Get product names
        top_k_names = [
            [product_names[idx] for idx in customer_indices]
            for customer_indices in top_k_indices
        ]

        logger.info(f'DONE: Top-K recommendations for {n_customers} customers completed')
        
        return top_k_indices, top_k_probas, top_k_names
    
    def recommend(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        k: int = TOP_K,
        return_probas: bool = True,
        apply_thresholds: bool = True
    ) -> pd.DataFrame:
        '''
            Generate product recommendations for customers.
        '''

        # Get top-k products for each customer
        top_k_indices, top_k_probas, top_k_names = self.predict_top_k(
            X, k, apply_thresholds
        )
        
        # Build recommendations
        recommendations = []
        for i in range(len(top_k_names)): # Loop over customers 
            rec = {
                'customer_idx': i,
                'recommendations': top_k_names[i]
            }
            if return_probas: # If return probabilities, add them to the recommendation
                rec['probabilities'] = top_k_probas[i].tolist()
            recommendations.append(rec)
        
        logger.info(f'DONE: Recommendations for {len(recommendations)} customers completed')

        return pd.DataFrame(recommendations)
    

    # ----------- Evaluation ------------ #
    def _calculate_map_at_k(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        '''
            Calculate Mean Average Precision @ K.
        '''
        aps = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx]
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Skip if no positive labels
            if true_labels.sum() == 0:
                continue
            
            # Get top-k indices
            top_k_indices = np.argsort(-pred_proba)[:self.top_k]
            
            hits = 0
            precision_sum = 0
            for i, idx_pred in enumerate(top_k_indices): # Loop over top-k indices
                if true_labels[idx_pred] == 1: # If true label is 1, increment hits
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            # Number of relevant items
            n_relevant = min(self.top_k, int(true_labels.sum()))
            # Average precision
            ap = precision_sum / n_relevant if n_relevant > 0 else 0
            # Append average precision to list
            aps.append(ap)
        
        logger.info(f'DONE: MAP@K calculation completed')

        return np.mean(aps) if aps else 0.0
    
    def _calculate_precision_at_k(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> float:
        '''
            Calculate Precision @ K.
        '''
        precisions = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx] # True labels for this sample
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Get top-k indices
            top_k_indices = np.argsort(-pred_proba)[:self.top_k]
            # Count hits
            hits = sum(true_labels[i] for i in top_k_indices)
            # Append precision to list
            precisions.append(hits / self.top_k)
        
        logger.info(f'DONE: Precision@K calculation completed')
        
        return np.mean(precisions) if precisions else 0.0
    
    def _calculate_precision_at_k_with_thresholds(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        products: List[str]
    ) -> float:
        '''
            Calculate Precision @ K with threshold filtering.
        '''
        precisions = []
        
        for idx in range(len(y_true)): # Loop over samples
            true_labels = y_true[idx] # True labels for this sample
            pred_proba = y_pred_proba[idx] # Predicted probabilities for this sample
            
            # Get sorted indices
            sorted_indices = np.argsort(-pred_proba)
            
            selected = []
            for i in sorted_indices: # Loop over sorted indices
                # If selected products are greater than or equal to k, break
                if len(selected) >= self.top_k: 
                    break
                product = products[i] # Get product name
                threshold = self.thresholds.get(product, 0.5) # Get threshold for this product
                # If probability is greater than threshold, add to selected
                if pred_proba[i] >= threshold:
                    selected.append(i)
            
             # If selected products are less than k, add remaining products
            if len(selected) < self.top_k:
                selected = list(sorted_indices[:self.top_k])
            
            # Count hits
            hits = sum(true_labels[i] for i in selected)
            # Append precision to list
            precisions.append(hits / self.top_k)
        
        logger.info(f'DONE: Precision@K with thresholds calculation completed')
        
        return np.mean(precisions) if precisions else 0.0

    def evaluate(
        self,
        X_test: Union[pl.DataFrame, pd.DataFrame],
        y_test: Union[pl.DataFrame, pd.DataFrame]
    ) -> Dict[str, Any]:
        '''
            Evaluate model performance on test data.
        '''

        logger.info('Evaluating OvR Grouped model')
        
        # Convert data to pandas DataFrames if needed
        X_pd, y_pd = self._prepare_data(X_test, y_test)
        
        # Get probabilities for all products
        y_pred_proba = self.predict_proba(X_pd)
        
        # Align columns
        # Get common products
        common_products = [p for p in self.all_products if p in y_pd.columns and p in y_pred_proba.columns]
        # Get true labels and probabilities for common products (fill nulls, convert to int)
        y_true_aligned = y_pd[common_products].fillna(False).astype(int).values # True labels
        y_proba_aligned = y_pred_proba[common_products].values # Predicted probabilities
        
        # Per-product metrics
        metrics_per_product = {}
        for i, product in enumerate(common_products): # Loop over common products
            y_true_i = y_true_aligned[:, i] # True labels for this product
            y_proba_i = y_proba_aligned[:, i] # Predicted probabilities for this product
            
            if y_true_i.sum() == 0: # If no positive cases, skip
                continue
            
            # Get threshold for this product
            threshold = self.thresholds.get(product, 0.5) 
            # Convert probabilities to binary predictions
            y_pred_i = (y_proba_i >= threshold).astype(int)
            
            # Calculate metrics for this product
            metrics_per_product[product] = {
                'auc_roc': roc_auc_score(y_true_i, y_proba_i),
                'avg_precision': average_precision_score(y_true_i, y_proba_i),
                'precision': precision_score(y_true_i, y_pred_i, zero_division=0),
                'recall': recall_score(y_true_i, y_pred_i, zero_division=0),
                'f1': f1_score(y_true_i, y_pred_i, zero_division=0),
                'log_loss': log_loss(y_true_i, y_proba_i),
                'positive_rate': float(y_true_i.mean()),
                'threshold': threshold
            }
        
        # Aggregate metrics
        aucs = [m['auc_roc'] for m in metrics_per_product.values()]
        avg_precs = [m['avg_precision'] for m in metrics_per_product.values()]
        
        # Mean Average Precision @ K - Average of precision at each rank
        map_at_k = self._calculate_map_at_k(y_true_aligned, y_proba_aligned)
        
        # Precision @ K - Ratio of true positives at each rank
        precision_at_k = self._calculate_precision_at_k(y_true_aligned, y_proba_aligned)
        
        # Precision @ K with thresholds - Ratio of true positives at each rank with threshold filtering
        precision_at_k_thresh = self._calculate_precision_at_k_with_thresholds(
            y_true_aligned, y_proba_aligned, common_products
        )
        
        # Overall metrics
        overall_metrics = {
            'mean_auc': np.mean(aucs) if aucs else 0.0,
            'std_auc': np.std(aucs) if aucs else 0.0,
            'mean_avg_precision': np.mean(avg_precs) if avg_precs else 0.0,
            f'map_at_{self.top_k}': map_at_k,
            f'precision_at_{self.top_k}': precision_at_k,
            f'precision_at_{self.top_k}_with_thresholds': precision_at_k_thresh,
            'n_products_evaluated': len(metrics_per_product)
        }
        
        # Group-level metrics
        group_metrics = {}
        for group_name, products in self.product_groups.items(): # Loop over groups
            group_products = [p for p in products if p in metrics_per_product] # Get products that are in this group and have metrics
            if group_products: # If there are products in this group
                group_aucs = [metrics_per_product[p]['auc_roc'] for p in group_products] # Get AUCs for this group
                group_metrics[group_name] = { # Store metrics for this group
                    'mean_auc': np.mean(group_aucs),
                    'n_products': len(group_products)
                }
        
        results = {
            'overall': overall_metrics,
            'per_group': group_metrics,
            'per_product': metrics_per_product
        }
        
        logger.info('DONE: Evaluation completed')
        logger.info(f"   Mean AUC: {overall_metrics['mean_auc']:.4f} +/- {overall_metrics['std_auc']:.4f}")
        logger.info(f'   MAP@{self.top_k}: {map_at_k:.4f}')
        logger.info(f'   Precision@{self.top_k}: {precision_at_k:.4f}')
        logger.info(f'   Precision@{self.top_k} (with thresholds): {precision_at_k_thresh:.4f}')
        
        for group_name, g_metrics in group_metrics.items():
            logger.info(f'DONE: Evaluation done for group: {group_name}')
            logger.info(f"{group_name}: AUC={g_metrics['mean_auc']:.4f} ({g_metrics['n_products']} products)")
        
        # Save results locally
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'ovr_grouped_evaluation_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'   Evaluation results saved to: {results_file}')
        
        return results
    
    # ----------- Results Access ------------ #
    def get_group_model(
        self, 
        group_name: str
    ) -> OneVsRestClassifier:
        '''
            Get the trained OvR model for a specific group.
        '''

        # If group not found, raise error
        if group_name not in self.models:
            raise ValueError(f'Group {group_name} not found.')

        return self.models[group_name]
    
    def get_group_products(
        self, group_name: str) -> List[str]:
        '''
            Get list of products in a specific group.
        '''

        # If group not found, raise error
        if group_name not in self.product_groups:
            raise ValueError(f'Group {group_name} not found.')

        return self.product_groups[group_name]
    
    def get_feature_importance(
        self,
        group_name: Optional[str] = None,
        top_n: int = 20
    ) -> pd.DataFrame:
        '''
            Get feature importance, optionally for a specific group.
        '''

        # If no models trained, raise error
        if not self.models:
            raise ValueError('No models trained.')
        
        # Get groups to use
        if group_name is not None: # If group name is provided, use only that group
            if group_name not in self.models: # If group not found, raise error
                raise ValueError(f'Group {group_name} not found')
            groups_to_use = {group_name: self.models[group_name]}
            products_to_use = {group_name: self.product_groups[group_name]}
        else:
            groups_to_use = self.models
            products_to_use = self.product_groups
        
        importance_data = []
        for g_name, model in groups_to_use.items(): # Loop over groups
            products = products_to_use[g_name] # Get products for this group
            for product, estimator in zip(products, model.estimators_): # Loop over products and estimators
                importance = estimator.get_feature_importance(type='FeatureImportance') # Get feature importance
                for feat_name, imp_value in zip(self.feature_names, importance): # Loop over feature names and importances
                    importance_data.append({
                        'group': g_name,
                        'product': product,
                        'feature': feat_name,
                        'importance': imp_value
                    })
        
        df_imp = pd.DataFrame(importance_data)
        
        mean_imp = (
            df_imp
                .groupby('feature')['importance']
                .mean()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
        )
        mean_imp.columns = ['feature', 'mean_importance']
        
        return mean_imp
    
    # ----------- Save / Load ------------ #
    def save(
        self, 
        model_name: str = 'ovr_grouped'
    ) -> str:
        '''
            Save all group models and metadata.
        '''

        # If no models trained, raise error
        if not self.models:
            raise ValueError('No models to save. Call fit() first.')
        
        # Create model directory if it doesn't exist
        model_dir = self.models_dir / f'{model_name}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each group model
        for group_name in tqdm(self.models.keys(), desc='Saving groups'):            
            model = self.models[group_name]
            products = self.product_groups[group_name]
            
            # Save estimators
            for product, estimator in zip(products, model.estimators_):
                estimator.save_model(str(model_dir / f'{group_name}_{product}.cbm'))
            
            # Save group metadata
            group_meta = {
                'products': products,
                'best_params': self.best_params.get(group_name, {}),
                'thresholds': {p: self.thresholds.get(p, 0.5) for p in products}
            }
            with open(model_dir / f'{group_name}_metadata.json', 'w') as f:
                json.dump(group_meta, f, indent=2)
        
        # Save main metadata
        metadata = {
            'all_products': self.all_products,
            'product_groups': self.product_groups,
            'thresholds': self.thresholds,
            'best_params': self.best_params,
            'feature_names': self.feature_names,
            'cat_features': self.cat_features,
            'cat_feature_indices': self.cat_feature_indices,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state,
            'n_splits': self.n_splits,
            'frequent_threshold': self.frequent_threshold,
            'rare_threshold': self.rare_threshold,
            'n_groups': len(self.models)
        }
        
        with open(model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'DONE: Models saved to: {model_dir}')

        return str(model_dir)
    
    def load(
        self, 
        model_path: str
    ) -> 'OvRGroupModel':
        '''
            Load group models from disk.
        '''
        model_dir = Path(model_path)
        
        # If model directory not found, raise error
        if not model_dir.exists():
            raise ValueError(f'Model directory not found: {model_path}')
        
        # Load main metadata
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.all_products = metadata['all_products']
        self.product_groups = metadata['product_groups']
        self.thresholds = metadata['thresholds']
        self.best_params = metadata.get('best_params', {})
        self.feature_names = metadata['feature_names']
        self.cat_features = metadata.get('cat_features', [])
        self.cat_feature_indices = metadata.get('cat_feature_indices', [])
        self.cv_scores = metadata.get('cv_scores', {})
        
        # Load each group
        self.models = {}
        
        for group_name in tqdm(['frequent', 'mid', 'rare'], desc='Loading groups'):
            # Load group metadata
            with open(model_dir / f'{group_name}_metadata.json', 'r') as f:
                group_meta = json.load(f)
            
            products = group_meta['products']
            
            # Load estimators
            estimators = []
            for product in products:
                model = CatBoostClassifier()
                model.load_model(str(model_dir / f'{group_name}_{product}.cbm'))
                estimators.append(model)
            
            # Reconstruct OvR
            ovr = OneVsRestClassifier(CatBoostClassifier())
            ovr.estimators_ = estimators
            ovr.classes_ = np.array([0, 1])
            
            self.models[group_name] = ovr
        
        logger.info(f'Loaded {len(self.models)} group models from: {model_path}')
        return self

def run_modelling_ovr():
    '''
        Run OvR Group Model training pipeline.
    '''
    logger.info('Starting OvR Group Model training pipeline')
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data(DATA_DIR)
    
    # Get product names
    target_names = get_product_names(y_train)
    
    # Detect categorical features
    cat_features = [
        col for col in X_train.columns
        if X_train[col].dtype == 'category' or X_train[col].dtype == 'object'
    ]
    
    # Initialize model
    model = OvRGroupModel(
        cat_features=cat_features,
        random_state=RANDOM_STATE,
        n_splits=N_SPLITS,
        mlflow_experiment=MLFLOW_EXPERIMENT,
        frequent_threshold=FREQUENT_THRESHOLD,
        rare_threshold=RARE_THRESHOLD,
        optimize=OPTIMIZE,
        n_trials=N_TRIALS,
        optuna_timeout=OPTUNA_TIMEOUT,
        run_cv=RUN_CV,
        top_k=TOP_K
    )
    
    # Train all group models
    model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        target_names=target_names
    )
    
    # Evaluate on test set
    results = model.evaluate(X_test, y_test)
    
    # Get top-7 recommendations for test customers
    recommendations = model.recommend(X_test)
    
    # Show sample recommendations
    logger.info('\nSample recommendations (first 5 customers):')
    for i in range(min(5, len(recommendations))):
        rec = recommendations.iloc[i]
        logger.info(f'  Customer {i}: {rec["recommendations"][:3]}...')
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    logger.info('Top 10 features mean importance:')
    logger.info(importance.to_string(index=False))
    
    # Save models
    model_path = model.save('ovr_grouped_santander')
    logger.info(f'\nModels saved to: {model_path}')
    
    # Example: Access specific group model
    # frequent_model = model.get_group_model('frequent')
    # frequent_products = model.get_group_products('frequent')
    
    # Example: Load saved models
    # loaded = OvRGroupModel().load(model_path)
    # new_recs = loaded.recommend(X_test, k=7)
    
    logger.info('OvR Group Model training pipeline completed successfully')


# ---------- Main ---------- #
if __name__ == '__main__':
    run_modelling_ovr()

# ---------- All exports ---------- #
__all__ = ['run_modelling_ovr', 'OvRGroupModel', 'load_training_data', 'get_product_names'] 
