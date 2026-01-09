'''
    CatBoost Modelling with TimeSeriesSplit & Optuna

    This module provides a CatBoostClassifier wrapper class with:
    - TimeSeriesSplit 5-fold cross-validation per product
    - Optuna hyperparameter optimization (per product)
    - MLflow logging for experiment tracking
    - Production-ready product → model mapping
    - Inference: rank top-7 products per customer

    Input:
    - X_train, y_train - Training features and targets (Polars DataFrames)
    - X_test, y_test - Test features and targets (Polars DataFrames)

    Output:
    - Trained models with optimized hyperparameters (24 products)
    - Evaluation metrics
    - Top-7 product recommendations per customer

    Usage:
    python -m src.modelling_catboost
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

from catboost import CatBoostClassifier, Pool
import optuna
from optuna.samplers import TPESampler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    average_precision_score,
    log_loss
)
from dotenv import load_dotenv
from tqdm import tqdm

# MLflow imports
try:
    import mlflow
    import mlflow.catboost
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from src.logging_setup import setup_logging

# ---------- Logging setup ---------- #
logger = setup_logging('modelling_catboost')

# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent
os.chdir(PROJECT_ROOT)

# ---------- Constants ---------- #
DATA_DIR = os.getenv('DATA_DIR', './data')
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
MODELS_DIR = os.getenv('MODELS_DIR', './models')

# Number of products (Santander dataset has 24 target products)
N_PRODUCTS = 24
# Number of CV folds for TimeSeriesSplit
N_SPLITS = 5
# Top-K recommendations
TOP_K = 7


# ---------- CatBoost Product Model Class ---------- #
class CatBoostProductModels:
    '''
        CatBoost classifier ensemble for multi-product recommendation.
        
        Trains one CatBoostClassifier per product (24 total) with:
        - TimeSeriesSplit 5-fold cross-validation
        - Optuna hyperparameter optimization
        - MLflow experiment tracking
    '''
    
    def __init__(
        self,
        cat_features: Optional[List[str]] = None,
        random_state: int = 42,
        n_splits: int = N_SPLITS,
        models_dir: str = MODELS_DIR,
        results_dir: str = RESULTS_DIR,
        mlflow_experiment: Optional[str] = 'santander_catboost'
    ):
        '''
            Initialize CatBoost product models.
            
            Args:
                cat_features: List of categorical feature names
                random_state: Random seed for reproducibility
                n_splits: Number of TimeSeriesSplit folds
                models_dir: Directory to save/load models
                results_dir: Directory to save results
                mlflow_experiment: MLflow experiment name (None to disable)
        '''
        self.cat_features = cat_features or []
        self.random_state = random_state
        self.n_splits = n_splits
        self.models_dir = Path(models_dir)
        self.results_dir = Path(results_dir)
        self.mlflow_experiment = mlflow_experiment
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Product → Model mapping (production registry)
        self.models: Dict[str, CatBoostClassifier] = {}
        
        # Best hyperparameters per product
        self.best_params: Dict[str, Dict[str, Any]] = {}
        
        # CV scores per product
        self.cv_scores: Dict[str, List[float]] = {}
        
        # Feature and target names
        self.feature_names: List[str] = []
        self.target_names: List[str] = []
        
        # Setup MLflow
        if MLFLOW_AVAILABLE and mlflow_experiment:
            mlflow.set_experiment(mlflow_experiment)
            logger.info(f'MLflow experiment set to: {mlflow_experiment}')
        
        logger.info(f'CatBoostProductModels initialized')
        logger.info(f'  - Random state: {random_state}')
        logger.info(f'  - CV splits: {n_splits}')
    
    def _prepare_data(
        self, 
        X: Union[pl.DataFrame, pd.DataFrame, np.ndarray],
        y: Optional[Union[pl.DataFrame, pd.DataFrame, np.ndarray]] = None
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        '''
            Convert input data to pandas DataFrames for CatBoost.
        '''
        # Convert X
        if isinstance(X, pl.DataFrame):
            X_pd = X.to_pandas()
        elif isinstance(X, np.ndarray):
            X_pd = pd.DataFrame(X, columns=self.feature_names if self.feature_names else None)
        else:
            X_pd = X.copy()
        
        # Convert y
        y_pd = None
        if y is not None:
            if isinstance(y, pl.DataFrame):
                y_pd = y.to_pandas()
            elif isinstance(y, np.ndarray):
                y_pd = pd.DataFrame(y)
            else:
                y_pd = y.copy()
        
        return X_pd, y_pd
    
    def _get_cat_feature_indices(self, X: pd.DataFrame) -> List[int]:
        '''
            Get indices of categorical features.
        '''
        if not self.cat_features:
            # Auto-detect categorical columns
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if X[col].dtype == 'object' or X[col].dtype.name == 'category'
            ]
        else:
            cat_indices = [
                i for i, col in enumerate(X.columns)
                if col in self.cat_features
            ]
        return cat_indices
    
    def _create_optuna_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cat_feature_indices: List[int],
        n_splits: int = 5
    ):
        '''
            Create Optuna objective with TimeSeriesSplit CV.
            
            Uses reduced search space and early stopping for faster optimization.
        '''
        def objective(trial: optuna.Trial) -> float:
            # Reduced hyperparameter search space for speed
            params = {
                'iterations': trial.suggest_int('iterations', 100, 500, step=100),
                'depth': trial.suggest_int('depth', 4, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.2, log=True),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                # Fixed parameters
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'random_seed': self.random_state,
                'verbose': False,
                'thread_count': -1,
                'auto_class_weights': 'Balanced',
            }
            
            # TimeSeriesSplit cross-validation
            tscv = TimeSeriesSplit(n_splits=n_splits)
            auc_scores = []
            
            for train_idx, val_idx in tscv.split(X_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                # Create pools
                train_pool = Pool(
                    X_tr, y_tr,
                    cat_features=cat_feature_indices if cat_feature_indices else None
                )
                val_pool = Pool(
                    X_val, y_val,
                    cat_features=cat_feature_indices if cat_feature_indices else None
                )
                
                # Train with early stopping
                model = CatBoostClassifier(**params)
                model.fit(
                    train_pool,
                    eval_set=val_pool,
                    early_stopping_rounds=30,
                    verbose=False
                )
                
                # Evaluate
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                
                # Handle edge cases
                if y_val.sum() == 0 or y_val.sum() == len(y_val):
                    continue
                    
                auc = roc_auc_score(y_val, y_pred_proba)
                auc_scores.append(auc)
            
            return np.mean(auc_scores) if auc_scores else 0.0
        
        return objective
    
    def _optimize_for_product(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        product_name: str,
        cat_feature_indices: List[int],
        n_trials: int = 15,
        timeout: int = 120
    ) -> Dict[str, Any]:
        '''
            Run Optuna optimization for a single product.
            
            Args:
                X_train: Training features
                y_train: Training target (single product)
                product_name: Name of the product
                cat_feature_indices: Categorical feature indices
                n_trials: Maximum Optuna trials
                timeout: Maximum optimization time (seconds)
            
            Returns:
                Best hyperparameters
        '''
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        # Create objective
        objective = self._create_optuna_objective(
            X_train, y_train, cat_feature_indices, self.n_splits
        )
        
        # Suppress Optuna logs
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # Optimize with timeout for faster execution
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=False
        )
        
        # Build full params dict
        best_params = study.best_params
        best_params.update({
            'loss_function': 'Logloss',
            'eval_metric': 'AUC',
            'random_seed': self.random_state,
            'verbose': False,
            'auto_class_weights': 'Balanced',
            'thread_count': -1
        })
        
        return best_params, study.best_value
    
    def _train_product_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.Series],
        product_name: str,
        cat_feature_indices: List[int],
        params: Dict[str, Any]
    ) -> CatBoostClassifier:
        '''
            Train a single product model with given hyperparameters.
        '''
        # Create pools
        train_pool = Pool(
            X_train, y_train.astype(int),
            cat_features=cat_feature_indices if cat_feature_indices else None
        )
        
        val_pool = None
        if X_val is not None and y_val is not None:
            val_pool = Pool(
                X_val, y_val.astype(int),
                cat_features=cat_feature_indices if cat_feature_indices else None
            )
        
        # Train model
        model = CatBoostClassifier(**params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50 if val_pool else None,
            verbose=False
        )
        
        return model
    
    def _cv_evaluate_product(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cat_feature_indices: List[int],
        params: Dict[str, Any]
    ) -> List[float]:
        '''
            Perform TimeSeriesSplit CV evaluation for a product.
            
            Returns list of AUC scores for each fold.
        '''
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        fold_scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Skip if validation set has no positive samples
            if y_val.sum() == 0:
                continue
            
            # Train
            train_pool = Pool(
                X_tr, y_tr.astype(int),
                cat_features=cat_feature_indices if cat_feature_indices else None
            )
            val_pool = Pool(
                X_val, y_val.astype(int),
                cat_features=cat_feature_indices if cat_feature_indices else None
            )
            
            model = CatBoostClassifier(**params)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=30, verbose=False)
            
            # Evaluate
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, y_pred_proba)
            fold_scores.append(auc)
        
        return fold_scores
    
    def fit(
        self,
        X_train: Union[pl.DataFrame, pd.DataFrame],
        y_train: Union[pl.DataFrame, pd.DataFrame],
        X_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        y_val: Optional[Union[pl.DataFrame, pd.DataFrame]] = None,
        target_names: Optional[List[str]] = None,
        optimize: bool = True,
        n_trials: int = 15,
        optuna_timeout: int = 120,
        run_cv: bool = True
    ) -> 'CatBoostProductModels':
        '''
            Train CatBoost models for all products (24 models).
            
            Args:
                X_train: Training features
                y_train: Training targets (24 product columns)
                X_val: Validation features (optional)
                y_val: Validation targets (optional)
                target_names: List of product/target names
                optimize: Run Optuna hyperparameter optimization
                n_trials: Optuna trials per product
                optuna_timeout: Timeout per product optimization (seconds)
                run_cv: Run 5-fold CV evaluation
            
            Returns:
                self
        '''
        logger.info('='*60)
        logger.info('Starting CatBoost Product Models Training')
        logger.info('='*60)
        
        # Prepare data
        X_pd, y_pd = self._prepare_data(X_train, y_train)
        self.feature_names = list(X_pd.columns)
        
        # Get categorical features
        cat_indices = self._get_cat_feature_indices(X_pd)
        logger.info(f'Features: {len(self.feature_names)}, Categorical: {len(cat_indices)}')
        
        # Get target names (products)
        if target_names is not None:
            self.target_names = target_names
        elif isinstance(y_pd, pd.DataFrame):
            self.target_names = list(y_pd.columns)
        else:
            self.target_names = [f'product_{i}' for i in range(y_pd.shape[1])]
        
        n_products = len(self.target_names)
        logger.info(f'Training {n_products} product models')
        
        # Prepare validation data
        X_val_pd, y_val_pd = None, None
        if X_val is not None and y_val is not None:
            X_val_pd, y_val_pd = self._prepare_data(X_val, y_val)
        
        # Start MLflow run
        mlflow_run = None
        if MLFLOW_AVAILABLE and self.mlflow_experiment:
            mlflow_run = mlflow.start_run(run_name=f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
            mlflow.log_param('n_products', n_products)
            mlflow.log_param('n_features', len(self.feature_names))
            mlflow.log_param('n_train_samples', len(X_pd))
            mlflow.log_param('optimize', optimize)
            mlflow.log_param('n_trials', n_trials)
            mlflow.log_param('n_cv_splits', self.n_splits)
        
        # Train each product model with progress bar
        product_progress = tqdm(
            self.target_names, 
            desc='Training Products',
            unit='product',
            ncols=100
        )
        
        all_cv_aucs = []
        
        try:
            for product_name in product_progress:
                product_progress.set_postfix({'current': product_name[:20]})
                
                # Get target column
                if isinstance(y_pd, pd.DataFrame):
                    y_product = y_pd[product_name]
                else:
                    y_product = y_pd
                
                # Get validation target if available
                y_val_product = None
                if y_val_pd is not None:
                    if isinstance(y_val_pd, pd.DataFrame):
                        y_val_product = y_val_pd[product_name]
                    else:
                        y_val_product = y_val_pd
                
                # Skip if no positive samples
                positive_rate = y_product.mean()
                if positive_rate == 0:
                    logger.warning(f'Skipping {product_name}: no positive samples')
                    continue
                
                # Start nested MLflow run for this product
                product_run = None
                if MLFLOW_AVAILABLE and mlflow_run:
                    product_run = mlflow.start_run(
                        run_name=f'product_{product_name}',
                        nested=True
                    )
                    mlflow.log_param('product_name', product_name)
                    mlflow.log_param('positive_rate', positive_rate)
                
                try:
                    # Hyperparameter optimization
                    if optimize:
                        best_params, best_auc = self._optimize_for_product(
                            X_pd, y_product, product_name, 
                            cat_indices, n_trials, optuna_timeout
                        )
                        self.best_params[product_name] = best_params
                        
                        if MLFLOW_AVAILABLE and product_run:
                            mlflow.log_params({f'hp_{k}': v for k, v in best_params.items() 
                                             if k not in ['verbose', 'thread_count']})
                            mlflow.log_metric('optuna_best_auc', best_auc)
                    else:
                        # Default parameters
                        best_params = {
                            'iterations': 300,
                            'depth': 6,
                            'learning_rate': 0.1,
                            'l2_leaf_reg': 3.0,
                            'loss_function': 'Logloss',
                            'eval_metric': 'AUC',
                            'random_seed': self.random_state,
                            'verbose': False,
                            'auto_class_weights': 'Balanced',
                            'thread_count': -1
                        }
                        self.best_params[product_name] = best_params
                    
                    # Run 5-fold CV evaluation
                    if run_cv:
                        cv_scores = self._cv_evaluate_product(
                            X_pd, y_product, cat_indices, best_params
                        )
                        self.cv_scores[product_name] = cv_scores
                        mean_cv_auc = np.mean(cv_scores) if cv_scores else 0.0
                        std_cv_auc = np.std(cv_scores) if cv_scores else 0.0
                        all_cv_aucs.append(mean_cv_auc)
                        
                        if MLFLOW_AVAILABLE and product_run:
                            mlflow.log_metric('cv_mean_auc', mean_cv_auc)
                            mlflow.log_metric('cv_std_auc', std_cv_auc)
                            for fold_idx, fold_auc in enumerate(cv_scores):
                                mlflow.log_metric(f'cv_fold_{fold_idx}_auc', fold_auc)
                    
                    # Train final model on full training data
                    model = self._train_product_model(
                        X_pd, y_product,
                        X_val_pd, y_val_product,
                        product_name, cat_indices, best_params
                    )
                    
                    # Store in product → model registry
                    self.models[product_name] = model
                    
                    # Log model to MLflow
                    if MLFLOW_AVAILABLE and product_run:
                        mlflow.catboost.log_model(model, f'model_{product_name}')
                    
                finally:
                    if MLFLOW_AVAILABLE and product_run:
                        mlflow.end_run()
            
            # Log overall metrics
            if MLFLOW_AVAILABLE and mlflow_run and all_cv_aucs:
                mlflow.log_metric('overall_mean_cv_auc', np.mean(all_cv_aucs))
                mlflow.log_metric('overall_std_cv_auc', np.std(all_cv_aucs))
                mlflow.log_metric('n_models_trained', len(self.models))
        
        finally:
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.end_run()
        
        logger.info('='*60)
        logger.info(f'Training completed: {len(self.models)} product models')
        if all_cv_aucs:
            logger.info(f'Overall CV AUC: {np.mean(all_cv_aucs):.4f} ± {np.std(all_cv_aucs):.4f}')
        logger.info('='*60)
        
        return self
    
    def predict_proba(
        self,
        X: Union[pl.DataFrame, pd.DataFrame]
    ) -> pd.DataFrame:
        '''
            Predict probabilities for all products.
            
            Args:
                X: Features (n_samples, n_features)
            
            Returns:
                DataFrame with probability for each product (n_samples, n_products)
        '''
        if not self.models:
            raise ValueError('No models trained. Call fit() first.')
        
        X_pd, _ = self._prepare_data(X)
        
        probas = {}
        for product_name, model in self.models.items():
            proba = model.predict_proba(X_pd)[:, 1]
            probas[product_name] = proba
        
        return pd.DataFrame(probas)
    
    def predict_top_k(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        k: int = TOP_K
    ) -> Tuple[np.ndarray, np.ndarray, List[List[str]]]:
        '''
            Get top-K product recommendations for each customer.
            
            Inference: runs customer through all 24 models, ranks by probability.
            
            Args:
                X: Features (n_customers, n_features)
                k: Number of top products to recommend
            
            Returns:
                Tuple of:
                - top_k_indices: (n_customers, k) indices of top products
                - top_k_probas: (n_customers, k) probabilities
                - top_k_names: List of lists with product names
        '''
        # Get probabilities from all models
        proba_df = self.predict_proba(X)
        proba_matrix = proba_df.values
        product_names = list(proba_df.columns)
        
        n_customers = proba_matrix.shape[0]
        
        # Rank products by probability for each customer
        top_k_indices = np.argsort(-proba_matrix, axis=1)[:, :k]
        
        # Get corresponding probabilities
        top_k_probas = np.take_along_axis(proba_matrix, top_k_indices, axis=1)
        
        # Get product names for each recommendation
        top_k_names = [
            [product_names[idx] for idx in customer_indices]
            for customer_indices in top_k_indices
        ]
        
        return top_k_indices, top_k_probas, top_k_names
    
    def recommend(
        self,
        X: Union[pl.DataFrame, pd.DataFrame],
        k: int = TOP_K,
        return_probas: bool = True
    ) -> pd.DataFrame:
        '''
            Generate product recommendations for customers.
            
            Args:
                X: Customer features
                k: Number of recommendations per customer
                return_probas: Include recommendation probabilities
            
            Returns:
                DataFrame with recommendations per customer
        '''
        top_k_indices, top_k_probas, top_k_names = self.predict_top_k(X, k)
        
        # Build recommendations DataFrame
        recommendations = []
        for i in range(len(top_k_names)):
            rec = {
                'customer_idx': i,
                'recommendations': top_k_names[i]
            }
            if return_probas:
                rec['probabilities'] = top_k_probas[i].tolist()
            recommendations.append(rec)
        
        return pd.DataFrame(recommendations)
    
    def evaluate(
        self,
        X_test: Union[pl.DataFrame, pd.DataFrame],
        y_test: Union[pl.DataFrame, pd.DataFrame],
        k: int = TOP_K
    ) -> Dict[str, Any]:
        '''
            Evaluate model performance on test data.
            
            Args:
                X_test: Test features
                y_test: Test targets
                k: K for MAP@K and Precision@K
            
            Returns:
                Evaluation metrics
        '''
        logger.info('Evaluating model performance')
        
        X_pd, y_pd = self._prepare_data(X_test, y_test)
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_pd)
        
        # Calculate per-product metrics
        metrics_per_product = {}
        for product_name in self.target_names:
            if product_name not in y_pd.columns or product_name not in y_pred_proba.columns:
                continue
            
            y_true = y_pd[product_name].astype(int)
            y_proba = y_pred_proba[product_name]
            
            # Skip if no positive samples
            if y_true.sum() == 0:
                continue
            
            # Compute metrics
            y_pred = (y_proba >= 0.5).astype(int)
            
            product_metrics = {
                'auc_roc': roc_auc_score(y_true, y_proba),
                'avg_precision': average_precision_score(y_true, y_proba),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'log_loss': log_loss(y_true, y_proba),
                'positive_rate': float(y_true.mean())
            }
            metrics_per_product[product_name] = product_metrics
        
        # Aggregate metrics
        aucs = [m['auc_roc'] for m in metrics_per_product.values()]
        avg_precs = [m['avg_precision'] for m in metrics_per_product.values()]
        
        # Calculate MAP@K
        map_at_k = self._calculate_map_at_k(y_pd, y_pred_proba, k)
        
        # Calculate Precision@K
        precision_at_k = self._calculate_precision_at_k(y_pd, y_pred_proba, k)
        
        overall_metrics = {
            'mean_auc': np.mean(aucs) if aucs else 0.0,
            'std_auc': np.std(aucs) if aucs else 0.0,
            'mean_avg_precision': np.mean(avg_precs) if avg_precs else 0.0,
            f'map_at_{k}': map_at_k,
            f'precision_at_{k}': precision_at_k,
            'n_products_evaluated': len(metrics_per_product)
        }
        
        results = {
            'overall': overall_metrics,
            'per_product': metrics_per_product
        }
        
        logger.info(f"Mean AUC: {overall_metrics['mean_auc']:.4f} ± {overall_metrics['std_auc']:.4f}")
        logger.info(f"MAP@{k}: {map_at_k:.4f}")
        logger.info(f"Precision@{k}: {precision_at_k:.4f}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'catboost_evaluation_{timestamp}.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f'Results saved to: {results_file}')
        
        return results
    
    def _calculate_map_at_k(
        self,
        y_true: pd.DataFrame,
        y_pred_proba: pd.DataFrame,
        k: int
    ) -> float:
        '''
            Calculate Mean Average Precision @ K.
        '''
        # Align columns
        common_cols = [c for c in y_true.columns if c in y_pred_proba.columns]
        y_true_aligned = y_true[common_cols]
        y_pred_aligned = y_pred_proba[common_cols]
        
        aps = []
        for idx in range(len(y_true_aligned)):
            true_labels = y_true_aligned.iloc[idx].values
            pred_proba = y_pred_aligned.iloc[idx].values
            
            # Skip if no positive labels
            if true_labels.sum() == 0:
                continue
            
            # Get top-k predictions
            top_k_indices = np.argsort(-pred_proba)[:k]
            
            # Calculate AP@K
            hits = 0
            precision_sum = 0
            for i, idx_pred in enumerate(top_k_indices):
                if true_labels[idx_pred] == 1:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            n_relevant = min(k, int(true_labels.sum()))
            ap = precision_sum / n_relevant if n_relevant > 0 else 0
            aps.append(ap)
        
        return np.mean(aps) if aps else 0.0
    
    def _calculate_precision_at_k(
        self,
        y_true: pd.DataFrame,
        y_pred_proba: pd.DataFrame,
        k: int
    ) -> float:
        '''
            Calculate Precision @ K.
        '''
        common_cols = [c for c in y_true.columns if c in y_pred_proba.columns]
        y_true_aligned = y_true[common_cols]
        y_pred_aligned = y_pred_proba[common_cols]
        
        precisions = []
        for idx in range(len(y_true_aligned)):
            true_labels = y_true_aligned.iloc[idx].values
            pred_proba = y_pred_aligned.iloc[idx].values
            
            top_k_indices = np.argsort(-pred_proba)[:k]
            hits = sum(true_labels[i] for i in top_k_indices)
            precision = hits / k
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def get_feature_importance(
        self,
        product_name: Optional[str] = None,
        importance_type: str = 'FeatureImportance',
        top_n: int = 20
    ) -> pd.DataFrame:
        '''
            Get feature importance, optionally for a specific product.
            
            Args:
                product_name: Specific product (None for average across all)
                importance_type: CatBoost importance type
                top_n: Number of top features to return
            
            Returns:
                DataFrame with feature importances
        '''
        if not self.models:
            raise ValueError('No models trained. Call fit() first.')
        
        if product_name is not None:
            if product_name not in self.models:
                raise ValueError(f'Model for {product_name} not found')
            models_to_use = {product_name: self.models[product_name]}
        else:
            models_to_use = self.models
        
        # Collect importances
        importance_data = []
        for p_name, model in models_to_use.items():
            importance = model.get_feature_importance(type=importance_type)
            for feat_name, imp_value in zip(self.feature_names, importance):
                importance_data.append({
                    'product': p_name,
                    'feature': feat_name,
                    'importance': imp_value
                })
        
        df_imp = pd.DataFrame(importance_data)
        
        # Calculate mean importance
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
    
    def get_product_model(self, product_name: str) -> CatBoostClassifier:
        '''
            Get the trained model for a specific product.
            
            Production access: product → model mapping.
            
            Args:
                product_name: Name of the product
            
            Returns:
                Trained CatBoostClassifier for that product
        '''
        if product_name not in self.models:
            raise ValueError(f'Model for {product_name} not found. Available: {list(self.models.keys())}')
        return self.models[product_name]
    
    def save(self, model_name: str = 'catboost_products') -> str:
        '''
            Save all product models and metadata.
            
            Args:
                model_name: Base name for saved files
            
            Returns:
                Path to saved model directory
        '''
        if not self.models:
            raise ValueError('No models to save. Call fit() first.')
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = self.models_dir / f'{model_name}_{timestamp}'
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save each product model
        logger.info(f'Saving {len(self.models)} product models...')
        for product_name, model in tqdm(self.models.items(), desc='Saving models'):
            model_path = model_dir / f'{product_name}.cbm'
            model.save_model(str(model_path))
        
        # Save metadata
        metadata = {
            'target_names': self.target_names,
            'feature_names': self.feature_names,
            'cat_features': self.cat_features,
            'best_params': self.best_params,
            'cv_scores': self.cv_scores,
            'random_state': self.random_state,
            'n_splits': self.n_splits,
            'n_models': len(self.models),
            'timestamp': timestamp
        }
        
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f'Models saved to: {model_dir}')
        return str(model_dir)
    
    def load(self, model_path: str) -> 'CatBoostProductModels':
        '''
            Load trained product models from disk.
            
            Args:
                model_path: Path to saved model directory
            
            Returns:
                self
        '''
        model_dir = Path(model_path)
        
        if not model_dir.exists():
            raise ValueError(f'Model directory not found: {model_path}')
        
        # Load metadata
        metadata_path = model_dir / 'metadata.json'
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.target_names = metadata['target_names']
        self.feature_names = metadata['feature_names']
        self.cat_features = metadata.get('cat_features', [])
        self.best_params = metadata.get('best_params', {})
        self.cv_scores = metadata.get('cv_scores', {})
        
        # Load each product model
        self.models = {}
        logger.info(f'Loading {metadata.get("n_models", len(self.target_names))} product models...')
        
        for product_name in tqdm(self.target_names, desc='Loading models'):
            model_file = model_dir / f'{product_name}.cbm'
            if model_file.exists():
                model = CatBoostClassifier()
                model.load_model(str(model_file))
                self.models[product_name] = model
        
        logger.info(f'Loaded {len(self.models)} models from: {model_path}')
        return self


# ---------- Convenience Functions ---------- #
def load_training_data(data_dir: str = DATA_DIR) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    '''
        Load train/test splits from parquet files.
    '''
    X_train = pl.read_parquet(f"{data_dir}/X_train.parquet")
    X_test = pl.read_parquet(f"{data_dir}/X_test.parquet")
    y_train = pl.read_parquet(f"{data_dir}/y_train.parquet")
    y_test = pl.read_parquet(f"{data_dir}/y_test.parquet")
    
    logger.info(f'Loaded data from: {data_dir}')
    logger.info(f'  X_train: {X_train.shape}, X_test: {X_test.shape}')
    logger.info(f'  y_train: {y_train.shape}, y_test: {y_test.shape}')
    
    return X_train, X_test, y_train, y_test


def get_product_names(y_train: pl.DataFrame) -> List[str]:
    '''
        Extract product names from target columns.
    '''
    return [col for col in y_train.columns if col.startswith('target_')]


# ---------- Main ---------- #
if __name__ == '__main__':
    logger.info('='*60)
    logger.info('CatBoost Product Models - Training Pipeline')
    logger.info('='*60)
    
    # Load data
    X_train, X_test, y_train, y_test = load_training_data(DATA_DIR)
    
    # Get product names (24 products)
    target_names = get_product_names(y_train)
    logger.info(f'Products to train: {len(target_names)}')
    
    # Detect categorical features
    cat_features = [
        col for col in X_train.columns
        if X_train[col].dtype == pl.Categorical or X_train[col].dtype == pl.Utf8
    ]
    logger.info(f'Categorical features: {len(cat_features)}')
    
    # Initialize model
    model = CatBoostProductModels(
        cat_features=cat_features,
        random_state=42,
        n_splits=5,
        mlflow_experiment='santander_catboost'
    )
    
    # Train all product models
    model.fit(
        X_train, y_train,
        X_val=X_test, y_val=y_test,
        target_names=target_names,
        optimize=True,
        n_trials=15,           # Reduced trials for speed
        optuna_timeout=120,    # 2 min timeout per product
        run_cv=True            # 5-fold TimeSeriesSplit CV
    )
    
    # Evaluate on test set
    results = model.evaluate(X_test, y_test, k=7)
    
    # Get top-7 recommendations for test customers
    logger.info('\nGenerating recommendations for test customers...')
    recommendations = model.recommend(X_test, k=7)
    logger.info(f'Generated recommendations for {len(recommendations)} customers')
    
    # Show sample recommendations
    logger.info('\nSample recommendations (first 5 customers):')
    for i in range(min(5, len(recommendations))):
        rec = recommendations.iloc[i]
        logger.info(f'  Customer {i}: {rec["recommendations"][:3]}...')
    
    # Feature importance
    importance = model.get_feature_importance(top_n=10)
    logger.info('\nTop 10 features (mean importance across products):')
    logger.info(importance.to_string(index=False))
    
    # Save models
    model_path = model.save('catboost_santander')
    logger.info(f'\nModels saved to: {model_path}')
    
    # Example: Access specific product model
    # product_model = model.get_product_model('target_ind_cco_fin_ult1')
    
    # Example: Load saved models
    # loaded = CatBoostProductModels().load(model_path)
    # new_predictions = loaded.predict_top_k(X_test, k=7)
    
    logger.info('\n' + '='*60)
    logger.info('Training pipeline completed successfully!')
    logger.info('='*60)
