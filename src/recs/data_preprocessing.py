'''
    Data Preprocessing

    This module provides functionality to preprocess raw data 
    into a format suitable for model training.

    Strategy:
    - Encode categorical columns
    - Load data with mixed-type columns as Utf8 (strings) to avoid casting errors
    - Apply manual transformations for columns with letter encodings
    - Cast remaining columns to target types
    - Drop features with high proportion of missing values
    - Clip outliers (age, antigüedad)
    - Impute renta (20% missing) by segmento + nomprov median
    - Log transform renta

    Input:
    - raw_dir - Directory with raw data files (train_ver2.csv),
    - preprocessed_dir - Output directory for processed parquet files.

    Output:
    - data_preprocessed.parquet - Preprocessed data
    - data_preprocessed_summary.parquet - Summary of preprocessed data

    Usage:
    python -m src.recs.data_preprocessing
'''

# ---------- Imports ---------- #
import os
import gc
import polars as pl
from dotenv import load_dotenv
from pathlib import Path
import json

from src.logging_setup import setup_logging


# ---------- Logging setup ---------- #
logger = setup_logging('data_preprocessing')


# ---------- Config ---------- #
# Load environment variables
load_dotenv()
# Set working directory to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent  # src/recs -> src -> project_root
os.chdir(PROJECT_ROOT)


# ---------- Constants ---------- #
# Data directory
DATA_DIR = os.getenv('DATA_DIR', './data')
# Results directory
RESULTS_DIR = os.getenv('RESULTS_DIR', './results')
# Raw data file
RAW_DATA_FILE = os.getenv('RAW_DATA_FILE', 'train_ver2.csv')
# Preprocessed data file
PREPROCESSED_DATA_FILE = os.getenv('PREPROCESSED_DATA_FILE', 'data_preprocessed.parquet')
# Preprocessed data summary file
PREPROCESSED_DATA_SUMMARY_FILE = os.getenv('PREPROCESSED_DATA_SUMMARY_FILE', 'data_preprocessed_summary.parquet')
# Encoding maps file
ENCODING_MAPS_FILE = os.getenv('ENCODING_MAPS_FILE', 'encoding_maps.json')
# App test data file
APP_TEST_DATA_FILE = os.getenv('APP_TEST_DATA_FILE', 'app_test_data.csv')
# App test preprocessed data file
APP_TEST_PREPROCESSED_DATA_FILE = os.getenv('APP_TEST_PREPROCESSED_DATA_FILE', 'app_test_data_preprocessed.parquet')


# ---------- Functions ---------- #
def load_and_encode_data(
    data_dir: str = DATA_DIR,
    results_dir: str = RESULTS_DIR,
    raw_file: str = RAW_DATA_FILE,
    encoding_maps_file: str = ENCODING_MAPS_FILE
) -> pl.DataFrame:
    '''
        Load CSV and auto-encode categorical columns during loading.
        
        Strategy:
        1. Load with mixed-type columns as Utf8 (strings) to avoid casting errors
        2. Apply manual transformations for columns with letter encodings
        3. Cast remaining columns to target types
    '''
       
    # Product columns (binary 0/1, but may have string issues)
    product_cols = [
        'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1',
        'ind_cder_fin_ult1', 'ind_cno_fin_ult1', 'ind_ctju_fin_ult1',
        'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1', 'ind_ctpp_fin_ult1',
        'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
        'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1',
        'ind_plan_fin_ult1', 'ind_pres_fin_ult1', 'ind_reca_fin_ult1',
        'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1', 'ind_viv_fin_ult1',
        'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1'
    ]
    
    # Load with safe schema (strings for problematic columns)
    load_schema = {
        # Dates - can be parsed directly
        'fecha_dato': pl.Utf8,
        'fecha_alta': pl.Utf8,
        'ult_fec_cli_1t': pl.Utf8,
        
        # Categoricals or ids - keep as strings initially
        'ncodpers': pl.Utf8,
        'ind_empleado': pl.Utf8,
        'pais_residencia': pl.Utf8,
        'canal_entrada': pl.Utf8,
        'nomprov': pl.Utf8,
        'indrel': pl.Utf8,
        
        # Mixed-type columns - must be loaded as strings first
        'sexo': pl.Utf8,
        'indrel_1mes': pl.Utf8,
        'tiprel_1mes': pl.Utf8,
        'conyuemp': pl.Utf8,
        'indfall': pl.Utf8,
        'segmento': pl.Utf8,
        
        # Numeric columns that might have string issues
        'age': pl.Utf8,
        'ind_nuevo': pl.Utf8,
        'indresi': pl.Utf8,
        'ind_actividad_cliente': pl.Utf8,
        'antiguedad': pl.Utf8,
        'indext': pl.Utf8,
        'tipodom': pl.Utf8,
        'cod_prov': pl.Utf8,
        'renta': pl.Utf8,
    }
    # Add binary product columns to schema
    for col in product_cols:
        load_schema[col] = pl.Utf8
    
    logger.info(f"Loading raw data from: {data_dir}/{raw_file}")
    df = pl.read_csv(
        f"{data_dir}/{raw_file}", 
        schema_overrides=load_schema,
        null_values=['NA', 'N/A', 'NaN', 'nan', 'null', 'None', '', ' '],
        ignore_errors=True
    )
    logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Define encoding mappings for letter-coded columns
    encoding_maps = {
        # sexo: H=male, V=female
        'sexo': {'H': 0, 'V': 1},
        # indrel: 1=primary client, 99=primary client this month, but not at the end of the month
        'indrel': {'1': 1, '99': 0},
        # ind_empleado: A=active, B=former employee, F=social leave, N=not employee, S=undefined status
        'ind_empleado': {'A': 1, 'B': 2, 'F': 3, 'N': 4, 'S': 5},
        # indrel_1mes: 1=premium, 2=owner, 3=former premium, P=potential
        # Note: P needs to be converted to numeric
        'indrel_1mes': {'1': 1, '1.0': 1, '2': 2, '2.0': 2, '3': 3, '3.0': 3, '4': 4, '4.0': 4, 'P': 5},       
        # tiprel_1mes: A=active, I=inactive, P=former, R=potential
        'tiprel_1mes': {'A': 1, 'I': 2, 'P': 3, 'R': 4, 'N': 5},        
        # conyuemp: S=yes, N=no
        'conyuemp': {'S': 1, 'N': 0},        
        # segmento: 01-TOP, 02-PARTICULARES, 03-UNIVERSITARIO
        'segmento': {'01 - TOP': 1, '02 - PARTICULARES': 2, '03 - UNIVERSITARIO': 3},
    }
    
    # Define boolean mappings (S/N or 1/0 strings)
    bool_maps = {
        # indfall: N=no, S=yes (deceased customer)
        'indfall': {'N': False, 'S': True, '0': False, '1': True},
        # ind_nuevo: 0=existing, 1=new
        'ind_nuevo': {'0': False, '1': True, '0.0': False, '1.0': True},        
        # indresi: S=resident, N=non-resident
        'indresi': {'S': True, 'N': False, '0': False, '1': True},
        # ind_actividad_cliente: 0=inactive, 1=active
        'ind_actividad_cliente': {'0': False, '1': True, '0.0': False, '1.0': True},
        # indext: S=foreigner, N=not foreigner
        'indext': {'S': 1, 'N': 0, '0': 0, '1': 1},
    }
    
    # Apply transformations
    logger.info('Applying data type transformations')
    
    # Transform date columns
    df = (
        df
            .with_columns([
                pl.col('fecha_dato').str.to_date('%Y-%m-%d').alias('fecha_dato'),
                pl.col('fecha_alta').str.to_date('%Y-%m-%d').alias('fecha_alta'),
                pl.col('ult_fec_cli_1t').str.to_date('%Y-%m-%d').alias('ult_fec_cli_1t'),
            ])
    )
    logger.info('Parsed date columns')

    # Transform categorical columns to Categorical
    # Strip whitespace before replacing
    for col, mapping in encoding_maps.items():
        if col in df.columns:
            df = (
                df
                    .with_columns(
                        pl.col(col).str.strip_chars().replace(mapping).cast(pl.Categorical).alias(col)
                    )
            )
    
    df = (
        df
            .with_columns([
                pl.col('cod_prov').str.strip_chars().cast(pl.Categorical),
                pl.col('pais_residencia').str.strip_chars().cast(pl.Categorical),
                pl.col('canal_entrada').str.strip_chars().cast(pl.Categorical),
                pl.col('nomprov').str.strip_chars().cast(pl.Categorical),
            ])
    )
    logger.info('Encoded categorical columns to Categorical')
    
    # Transform boolean columns using pl.when()
    # Strip whitespace first for all comparisons 
    df = (
        df
            # indfall: N=no, S=yes (deceased)
            .with_columns(
                pl.when(pl.col('indfall').str.strip_chars().is_in(['S', '1', '1.0']))
                .then(True)
                .when(pl.col('indfall').str.strip_chars().is_in(['N', '0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias('indfall')
            )
            # ind_nuevo: 0=existing, 1=new
            .with_columns(
                pl.when(pl.col('ind_nuevo').str.strip_chars().is_in(['1', '1.0']))
                .then(True)
                .when(pl.col('ind_nuevo').str.strip_chars().is_in(['0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias('ind_nuevo')
            )
            # indresi: S=resident, N=non-resident
            .with_columns(
                pl.when(pl.col('indresi').str.strip_chars().is_in(['S', '1', '1.0']))
                .then(True)
                .when(pl.col('indresi').str.strip_chars().is_in(['N', '0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias('indresi')
            )
            # ind_actividad_cliente: 0=inactive, 1=active
            .with_columns(
                pl.when(pl.col('ind_actividad_cliente').str.strip_chars().is_in(['1', '1.0']))
                .then(True)
                .when(pl.col('ind_actividad_cliente').str.strip_chars().is_in(['0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias('ind_actividad_cliente')
            )
            # indext: S=foreigner, N=not foreigner
            .with_columns(
                pl.when(pl.col('indext').str.strip_chars().is_in(['S', '1', '1.0']))
                .then(True)
                .when(pl.col('indext').str.strip_chars().is_in(['N', '0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias('indext')
            )
            # Transform product columns to Boolean
            .with_columns(
                pl.when(pl.col(col).str.strip_chars().is_in(['1', '1.0']))
                .then(True)
                .when(pl.col(col).str.strip_chars().is_in(['0', '0.0']))
                .then(False)
                .otherwise(None)
                .alias(col)
                for col in product_cols if col in df.columns
            )
    )
    logger.info('Encoded boolean and product columns to Boolean')
    
    # Transform numeric columns
    # Convert invalid values (NA, empty strings, etc.) to null
    df = (
        df
            .with_columns([
                pl.col('age').str.strip_chars().str.to_integer(strict=False).cast(pl.UInt8).alias('age'),
                # antiguedad has -999999 sentinel values for missing data - convert to null
                pl.when(pl.col('antiguedad').str.strip_chars().str.to_integer(strict=False) < 0)
                    .then(None)
                    .otherwise(pl.col('antiguedad').str.strip_chars().str.to_integer(strict=False))
                    .cast(pl.UInt32)
                    .alias('antiguedad')
            ])
            .with_columns(
                pl.when(
                    pl.col('renta').str.strip_chars()
                    .str.replace(',', '.')
                    .str.contains(r'^-?\d+\.?\d*$')
                )
                    .then(pl.col('renta').str.strip_chars().str.replace(',', '.').cast(pl.Float32))
                    .otherwise(None)
                .alias('renta')
            )
    )
    logger.info('Casted numeric columns to UInt8, UInt32 and Float32')
    
    # Log final schema
    logger.info('Final schema:')
    for col, dtype in zip(df.columns, df.dtypes):
        logger.info(f"  {col}: {dtype}")

    # Save encoding maps for reference
    all_maps = {**encoding_maps, **bool_maps}
    # Convert bool values to strings for JSON serialization
    serializable_maps = {
        k: {str(kk): str(vv) for kk, vv in v.items()} 
        for k, v in all_maps.items()
    }
    with open(f"{results_dir}/{encoding_maps_file}", 'w') as f:
        json.dump(serializable_maps, f, indent=2)
    logger.info(f"Encoding maps saved to: {results_dir}/{encoding_maps_file}")
    
    return df

def preprocess_data(
    df: pl.DataFrame, 
    data_dir: str = DATA_DIR, 
    results_dir: str = RESULTS_DIR,
    preprocessed_data_file: str = PREPROCESSED_DATA_FILE,
    preprocessed_data_summary_file: str = 'data_preprocessed_summary.parquet'
) -> pl.DataFrame:
    logger.info('Starting data preprocessing pipeline')
    
    # Create 'customer_period' feature: difference between fecha_dato and fecha_alta in months
    df_preprocessed = (
        df
            .with_columns([
                (
                    (pl.col('fecha_dato').dt.year() - pl.col('fecha_alta').dt.year()) * 12
                    + (pl.col('fecha_dato').dt.month() - pl.col('fecha_alta').dt.month())
                ).alias('customer_period')
            ])
    )
    logger.info('Computed customer_period (months since fecha_alta)')
    
    # Drop features with high proportion of missing values 'ult_fec_cli_1t', 'conyuemp'
    # Drop non-indicative feature 'tipodom', 'indext'
    # Drop 'fecha_alta' feature as CatBoost cannot handle dates
    # Drop 'antiguedad' feature as it is highly correlated with customer_period
    df_preprocessed = df_preprocessed.drop(['ult_fec_cli_1t', 'conyuemp', 'tipodom', 'fecha_alta', 'indext', 'antiguedad'])
    logger.info(f'Dropped features: {["ult_fec_cli_1t", "conyuemp", "tipodom", "fecha_alta", "indext", "antiguedad"]}')
    
    # Clip outliers
    df_preprocessed = (
        df_preprocessed
            .with_columns([
                pl.col('age').clip(18, 90).alias('age')
            ])
    )
    logger.info(f'Clipped outliers: {["age"]}')
    
    # Impute missing values in categorical columns
    # Fill with 'missing'
    categorical_cols = [col for col in df_preprocessed.columns if df_preprocessed[col].dtype == pl.Categorical]
    df_preprocessed = (
        df_preprocessed
            .with_columns([
                pl.col(col).cast(pl.Utf8).fill_null('missing').cast(pl.Categorical).alias(col)
                for col in categorical_cols
            ])
    )
    logger.info('Filled missing values in categorical columns with "missing"')
    
    # Impute renta missing values using median by customer segment and province
    renta_medians = (
        df_preprocessed
            .filter(pl.col('renta').is_not_null())
            .group_by(['segmento', 'nomprov'])
            .agg(pl.col('renta').median().alias('renta_median'))
    )
    df_preprocessed = (
        df_preprocessed
            .join(renta_medians, on=['segmento', 'nomprov'], how='left')
            .with_columns([
                pl.when(pl.col('renta').is_null())
                .then(pl.col('renta_median').fill_null(
                    df_preprocessed.filter(pl.col('renta').is_not_null())['renta'].median()
            ))
            .otherwise(pl.col('renta'))
            .alias('renta')
        ])
        .drop('renta_median')
    )
    logger.info('Imputed renta missing values')
    
    # Drop 'nomprov' (used for renta imputation, now no longer needed)
    df_preprocessed = df_preprocessed.drop(['nomprov'])
    logger.info(f'Dropped features: {["nomprov"]}')
   
    # Drop rows where sexo is null
    # As data contain rows with almost all features are null
    df_preprocessed = df_preprocessed.filter(pl.col('sexo').is_not_null())
    logger.info('Dropped rows with almost all features null')

    # Drop rows where ind_nomina_ult1 or ind_nom_pens_ult1 is null
    # As these features would be used for targets creation
    df_preprocessed = df_preprocessed.filter(pl.col('ind_nomina_ult1').is_not_null() & pl.col('ind_nom_pens_ult1').is_not_null())
    logger.info('Dropped rows where ind_nomina_ult1 or ind_nom_pens_ult1 is null')
    
    # Save preprocessed data
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    df_preprocessed.write_parquet(f"{data_dir}/{preprocessed_data_file}")
    logger.info(f"Preprocessed data saved to: {data_dir}/{preprocessed_data_file}")

    # Save preprocessed data summary
    if preprocessed_data_summary_file is not None:
        df_preprocessed_summary = df_preprocessed.describe()
        df_preprocessed_summary.write_parquet(f"{results_dir}/{preprocessed_data_summary_file}")
        logger.info(f"Preprocessed data summary saved to: {results_dir}/{preprocessed_data_summary_file}")
   
    del df_preprocessed_summary
    gc.collect()

    return df_preprocessed

def run_preprocessing():
    logger.info('Starting data preprocessing pipeline')
    df_encoded = load_and_encode_data(DATA_DIR, RESULTS_DIR, RAW_DATA_FILE, ENCODING_MAPS_FILE)
    df = preprocess_data(df_encoded, DATA_DIR, RESULTS_DIR, PREPROCESSED_DATA_FILE, PREPROCESSED_DATA_SUMMARY_FILE)
    del df_encoded, df
    gc.collect()
    logger.info('Data preprocessing completed successfully')


# ---------- Main function ---------- #
if __name__ == '__main__':
    run_preprocessing()


# ---------- All exports ---------- #
__all__ = ['run_preprocessing', 'load_and_encode_data', 'preprocess_data']