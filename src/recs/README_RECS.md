# Bank Products Modelling Pipeline
Overview of the ML pipeline for training the bank product recommendation model.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        Recommendation Pipeline (src/recs)                            │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   ┌──────────────────┐                                                              │
│   │ Step 1: Data     │  • Download raw data from Yandex.Disk                        │
│   │ Loading          │  • Extract ZIP archive                                       │
│   │ (data_loading.py)│  • Save train_ver2.csv to data/                              │
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 2: Data     │  • Load CSV with schema overrides                            │
│   │ Preprocessing    │  • Encode categorical variables                              │
│   │(data_preprocess  │  • Convert boolean columns                                   │
│   │    ing.py)       │  • Handle missing values (imputation)                        │
│   └────────┬─────────┘  • Create customer_period feature                            │
│            │            • Drop high-null columns                                     │
│            ▼            • Save data_preprocessed.parquet                            │
│   ┌──────────────────┐                                                              │
│   │ Step 3: Feature  │  • Add lag features (3m, 6m)                                 │
│   │ Engineering      │  • Add "acquired_recently" features                          │
│   │(feature_engineer │  • Add product interaction features                          │
│   │    ing.py)       │  • Update data_preprocessed.parquet                          │
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 4: Target   │  • Create 24 binary target columns                           │
│   │ Engineering      │  • Target = 1 if customer will add product next month        │
│   │(target_engineer  │  • Filter out last month (no targets)                        │
│   │    ing.py)       │  • Update data_preprocessed.parquet                          │
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 5: Train/   │  • All Positives + Random Negatives sampling                 │
│   │ Test Split       │  • Temporal split (last month - test)                        │
│   │(train_test_split │  • Save X_train, X_test, y_train, y_test                     │
│   │    .py)          │                                                              │
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 6: Model    │  • Group products by prevalence                              │
│   │ Training         │  • Optuna hyperparameter optimization per group on sampled data with CV evaluation                     │
│   │(modelling_ovr.py)│  • Train OvR CatBoost on sampled data                            │
│   │                  │  • Optimize thresholds per product
                         │• Retrain CatBoost for each product on full dataset
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 7: Model    │  • Evaluate on test set                                      │
│   │ Evaluation       │  • Calculate AUC, MAP@N, Precision@N                         │
│   │(modelling_ovr.py)│  • Save evaluation results                                   │
│   └────────┬─────────┘                                                              │
│            │                                                                         │
│            ▼                                                                         │
│   ┌──────────────────┐                                                              │
│   │ Step 8: Save &   │  • Save models to models/ovr_grouped_catboost/               │
│   │ Log to MLflow    │  • Log models, metrics, artifacts to MLflow                  │
│   │(mlflow_logging   │  • Register model in MLflow Model Registry                   │
│   │    .py)          │                                                              │
│   └──────────────────┘                                                              │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Step Descriptions

### Step 1: Data Loading (`data_loading.py`)
Purpose: Download raw data from Yandex.Disk cloud storage.
Input: Public Yandex.Disk URL
Output: `data/train_ver2.csv` - raw dataset
Process:
1. Get direct download link from Yandex.Disk API
2. Download file content (ZIP or CSV)
3. Extract ZIP if needed
4. Save to data directory
```python
# Run standalone
python -m src.recs.data_loading
```

### Step 2: Data Preprocessing (`data_preprocessing.py`)
Purpose: Clean and transform raw data for ML model consumption.
Input: `data/train_ver2.csv`
Output:  
- `data/data_preprocessed.parquet`
- `results/encoding_maps.json`
Process:
1. Load with safe schema - Handle mixed-type columns as strings
2. Parse dates - Convert fecha_dato, fecha_alta to Date type
3. Encode categoricals - Map letter codes to integers:
   - `sexo`: H→0, V→1
   - `ind_empleado`: A→1, B→2, F→3, N→4, S→5
   - `indrel_1mes`: 1→1, 2→2, 3→3, P→5
   - `tiprel_1mes`: A→1, I→2, P→3, R→4
   - `segmento`: 01-TOP→1, 02-PARTICULARES→2, 03-UNIVERSITARIO→3
4. Convert booleans - S/N strings to True/False
5. Convert products - 24 product columns to boolean
6. Create customer_period - Months since customer joined
7. Drop high-null columns - ult_fec_cli_1t, conyuemp, tipodom, etc.
8. Clip outliers - Age: 18-90
9. Impute renta - Fill missing income by segment+province median
10. Drop incomplete rows - Remove rows with null products
```python
# Run standalone
python -m src.recs.data_preprocessing
```

### Step 3: Feature Engineering (`feature_engineering.py`)
Purpose: Create additional features to capture temporal patterns.
Input: `data/data_preprocessed.parquet`
Output: Updated `data/data_preprocessed.parquet` with new features
Process:
1. Create Lag Features (3 and 6 months):
   - `ind_*_ult1_lag3` - Product ownership 3 months ago
   - `ind_*_ult1_lag6` - Product ownership 6 months ago  
   - `n_products_lag3` - Total products 3 months ago
   - `n_products_lag6` - Total products 6 months ago
2. Create Recently Acquired Features:
   - `ind_*_ult1_acquired_recently` - 1 if product was acquired in last month
3. Create Product Interaction Features:
   - `ind_nomina_ult1_ind_nom_pens_ult1_interaction`
   - `ind_cno_fin_ult1_ind_nom_pens_ult1_interaction`
   - `ind_cno_fin_ult1_ind_nomina_ult1_interaction`
   - `ind_cno_fin_ult1_ind_recibo_ult1_interaction`
   - `ind_nomina_ult1_ind_recibo_ult1_interaction`
```python
# Run standalone
python -m src.recs.feature_engineering
```

### Step 4: Target Engineering (`target_engineering.py`)
Purpose: Create binary target columns for each of the 24 products.
Input: `data/data_preprocessed.parquet`
Output: Updated `data/data_preprocessed.parquet` with 24 target columns
Process**:
1. For each product column `ind_*_ult1`:
   - Create `target_*` = 1 if customer doesn't have product now but will have it next month
   - Formula: `(ind_*_ult1_next_month == 1) & (ind_*_ult1_current == 0)`
2. Filter out last month rows (no future data for targets)
```python
# Run standalone
python -m src.recs.target_engineering
```

### Step 5: Train/Test Split (`train_test_split.py`)
Purpose: Create train and test datasets using temporal splitting.
Input: `data/data_preprocessed.parquet`
Output: 
- `data/X_train.parquet` - Training features
- `data/X_test.parquet` - Test features  
- `data/y_train.parquet` - Training targets
- `data/y_test.parquet` - Test targets
Process:
1. Sampling Strategy - All Positives + Random Negatives:
   - Keep all rows where at least one target = 1 (customer will add a product)
   - Randomly sample negative rows to reach ~10% of total data
2. Temporal Split:
   - Train: All months except the last
   - Test: Last month only (avoid data leakage)
3. Feature/Target Separation:
   - Features: All columns except IDs, targets, and datetime columns
   - Targets: All `target_*` columns
```python
# Run standalone
python -m src.recs.train_test_split
```

### Step 6: Model Training (`modelling_ovr.py`)
Purpose: Train One-vs-Rest CatBoost classifiers grouped by product prevalence.
Input: 
- `data/X_train.parquet`, `data/y_train.parquet`
- `data/X_test.parquet`, `data/y_test.parquet`
Output: 
- `models/ovr_grouped_catboost/` - Saved models
- `results/ovr_group_thresholds.json` - Optimized thresholds
Process:
1. Group Products by Prevalence:
   Frequent (>0.5%):  8 products - most common acquisitions
   Mid (0.01%-0.5%):  7 products - moderate frequency
   Rare (<0.01%):     9 products - least common
2. Hyperparameter Optimization (per group):
   - Optuna with TPE sampler
   - TimeSeriesSplit CV (3 folds)
   - Search space: iterations, depth, learning_rate, l2_leaf_reg
3. Train OvR Models:
   - One CatBoost binary classifier per product
   - Uses balanced class weights
   - Early stopping for efficiency
4. Threshold Optimization:
   - Find optimal probability threshold per product
   - Maximize F1 score on validation set
5. Retrain on Full Data:
   - Re-fit models using full preprocessed data with best hyperparameters
```python
# Run standalone
python -m src.recs.modelling_ovr
```

### Step 7: Model Evaluation (`modelling_ovr.py`)
Purpose: Evaluate model performance on test set.
Output: `results/ovr_grouped_evaluation_*.json`
Metrics Computed:
- AUC-ROC - Area under ROC curve (per product)
- Average Precision - Area under PR curve (per product)
- Precision - True positives / predicted positives
- Recall - True positives / actual positives
- F1 Score - Harmonic mean of precision & recall
- MAP@N - Mean Average Precision at N
- Precision@N - Precision considering only top-N recommendations

### Step 8: MLflow Logging (`mlflow_logging.py`)
Purpose: Log trained models and metadata to MLflow for tracking and deployment.
What to Log:
1. Parameters:
   - Number of products, features, groups
   - Prevalence thresholds
   - Best hyperparameters per group
2. Metrics:
   - CV scores per group
   - Overall mean AUC
3. Artifacts:
   - CatBoost models (24 total)
   - Thresholds JSON
   - Product groups JSON
   - Model metadata JSON
   - Feature importance CSV
4. Model Registry:
   - Register models for versioning
   - Enable model serving
```python
# Run standalone
python -m src.recs.mlflow_logging \
  --model-path models/ovr_grouped_catboost \
  --data-path data/X_test.parquet \
  --experiment bank_products_recommendation
```


## Running the Pipeline

### Full Pipeline

```bash
# Run all steps
python3 -m src.recs.main_recs
```

### With Options

```bash
# Skip data download (if data already exists)
python3 -m src.recs.main_recs --skip-download

# Skip modelling (only run preprocessing)
python3 -m src.recs.main_recs --skip-modelling
```
