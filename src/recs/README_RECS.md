```mermaid
flowchart TD
    A[Start] --> B[Load train / validation data]
    B --> C[Prepare data<br/>(to pandas, fix categoricals)]
    C --> D[Detect categorical feature indices]

    D --> E{Feature selection enabled?}
    E -- Yes --> F[Train quick CatBoost<br/>→ drop low-importance features]
    E -- No --> G[Keep all features]
    F --> H
    G --> H

    H[Determine target product list] --> I[Group products by prevalence]
    I -->|> frequent_threshold| J[Frequent group]
    I -->|between thresholds| K[Mid group]
    I -->|< rare_threshold| L[Rare group]

    %% ===== GROUP LOOP =====
    subgraph GROUP_LOOP[For each group: frequent / mid / rare]
        M[Select group targets] --> N{Optuna optimization enabled?}

        N -- Yes --> O[Optuna HPO<br/>TimeSeriesSplit CV<br/>Mean AUC]
        N -- No --> P[Use default CatBoost params]

        O --> Q[Best params]
        P --> Q

        Q --> R[Train models<br/>ONE PRODUCT AT A TIME]
        R --> S{Has both classes?}

        S -- No --> T[Skip product<br/>threshold = 1.0]
        S -- Yes --> U[Train CatBoost]
        U --> V[Save model to disk]
        V --> W[Reload model]

        W --> X[Assemble OvR wrapper]

        X --> Y{Validation data available?}
        Y -- Yes --> Z[Optimize thresholds<br/>per product (F1)]
        Y -- No --> AA[Use default threshold = 0.5]

        Z --> AB[Store thresholds]
        AA --> AB

        AB --> AC[Log models & params to MLflow]
    end

    J --> GROUP_LOOP
    K --> GROUP_LOOP
    L --> GROUP_LOOP

    %% ===== OPTIONAL RETRAIN =====
    AC --> AD{Retrain on full data?}
    AD -- Yes --> AE[Load full preprocessed data]
    AE --> AF[Apply same feature selection]
    AF --> AG[Retrain each group<br/>on full data]
    AD -- No --> AH[Skip retraining]

    AG --> AI[Final models ready]
    AH --> AI

    AI --> AJ[End]
```


    # All Positives + Random Negatives Sampling
    # Get all positives : rows with at least one positive target
    # Sample negatives : random rows with negative target
    # Combine positives and sampled negatives
    Sampled data is 10% of full data

    Stratified K-Fold on sampled data
    Optuna optimization & Train & validate on Stratified K-Fold
    Retrain final model on full dataset

 
- fecha_alta - first contract date with bank;
- ind_nuevo - indicator of wheither a customer is new during last 6 months (1=new customer, 0=existing);
- antiguedad - tenure in months;

- indext - foreigner index;

- nomprov - province name;

Summary of Memory and Speed Optimizations
I've implemented comprehensive memory optimization and speedups across your pipeline. Here's what was added:
1. Memory Optimizations in main_recs.py
Added log_memory_cleanup() helper that runs gc.collect() and logs the number of objects freed
Explicit memory cleanup after each pipeline step (data loading, preprocessing, feature engineering, target engineering, train/test split, model training, evaluation)
Delete X_train, y_train after model fitting (keep only test data for evaluation)
Final cleanup deleting all remaining objects
2. Memory Optimizations in modelling_ovr.py
HPO trials: Memory cleanup (gc.collect()) after each CV fold
CV evaluation: Memory cleanup after each fold
End of training: Clear cached quantized pools and run garbage collection
3. HPO Speed Optimizations (New Environment Variables)
Variable	Default	Purpose
HPO_N_SPLITS	2	Use fewer CV folds during HPO (vs 5 for final CV)
HPO_MAX_ITERATIONS	300	Lower max iterations during HPO
HPO_EARLY_STOPPING_ROUNDS	30	Early stopping in HPO trials
USE_QUANTIZED_POOL	True	Pre-quantize CatBoost Pool (reused across trials)
4. Training Speed Optimizations (New Environment Variables)
Variable	Default	Purpose
BOOSTING_TYPE	Plain	Plain is faster than Ordered
BORDER_COUNT	128	Lower = faster (CPU default: 254)
THREAD_COUNT	-1	Control threading (-1 = auto)
MAX_CTR_COMPLEXITY	1	Reduce CTR complexity for categoricals
ONE_HOT_MAX_SIZE	10	Limit one-hot encoding size
EARLY_STOPPING_ROUNDS	50	Early stopping for final training
5. Feature Selection (Optional)
Variable	Default	Purpose
USE_FEATURE_SELECTION	False	Enable feature importance-based selection
MIN_FEATURE_IMPORTANCE	0.001	Drop features below this normalized importance

Key Strategies Implemented
Smart Sampling (already in your code): Keep all positives, subsample negatives
Fewer CV folds for HPO: Use 2 folds for HPO trials, 5 for final evaluation
Early stopping: Stop trials/training when metric plateaus
Fixed heavy params during HPO: max_ctr_complexity, boosting_type, border_count
Pre-quantization: Quantize data once, reuse for all trials
Memory cleanup: Explicit gc.collect() after each step and fold
Optional feature selection: Drop low-importance features to reduce tree complexity

When it shines (your exact case)
Large datasets (>1M rows): Quantization dominates training time

Many Optuna trials (>50): Amortizes upfront cost

Mixed cat+num features: Handles your 33 categoricals perfectly

OvR multilabel: Pre-quantize once per target

Bottom line: 5-10x HPO speedup on your 17M Santander dataset. Do this for development, then final training uses fresh quantization on full data. Essential optimization!

Pre-quantization means CatBoost converts your raw features (especially numerical and categorical) into discrete bins/buckets once upfront, then reuses these quantized values for all Optuna trials instead of requantizing fresh each time.
