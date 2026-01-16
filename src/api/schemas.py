'''
    Pydantic Schemas for API Request/Response Validation

    This module defines the data schemas for the Bank Products Recommender API.
    All request and response models are validated using Pydantic.

    Usage:
    from src.api.schemas import PredictionRequest, PredictionResponse
'''

# ---------- Imports ---------- #
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any, Union
from datetime import date


# ---------- Feature Schema ---------- #
# Complete feature schema based on data_preprocessed.parquet
FEATURE_SCHEMA = {
    # Core identifiers
    'fecha_dato': {'type': 'date', 'required': True, 'description': 'Reference date'},
    'ncodpers': {'type': 'str', 'required': True, 'description': 'Customer ID'},
    
    # Categorical features
    'ind_empleado': {'type': 'str', 'required': True, 'description': 'Employee indicator'},
    'pais_residencia': {'type': 'str', 'required': True, 'description': 'Country of residence'},
    'sexo': {'type': 'str', 'required': True, 'description': 'Gender'},
    'indrel': {'type': 'str', 'required': True, 'description': 'Customer relationship type'},
    'indrel_1mes': {'type': 'str', 'required': True, 'description': 'Customer type at beginning of month'},
    'tiprel_1mes': {'type': 'str', 'required': True, 'description': 'Customer relation type at beginning of month'},
    'canal_entrada': {'type': 'str', 'required': True, 'description': 'Channel used to join'},
    'cod_prov': {'type': 'str', 'required': True, 'description': 'Province code'},
    'segmento': {'type': 'str', 'required': True, 'description': 'Customer segment'},
    
    # Numeric features
    'age': {'type': 'int', 'required': True, 'min': 0, 'max': 150, 'description': 'Customer age'},
    'renta': {'type': 'float', 'required': True, 'description': 'Household income'},
    'customer_period': {'type': 'int', 'required': True, 'description': 'Customer tenure period'},
    
    # Boolean features
    'ind_nuevo': {'type': 'bool', 'required': True, 'description': 'New customer indicator'},
    'indresi': {'type': 'bool', 'required': True, 'description': 'Residence indicator'},
    'indfall': {'type': 'bool', 'required': True, 'description': 'Deceased indicator'},
    'ind_actividad_cliente': {'type': 'bool', 'required': True, 'description': 'Activity indicator'},
    
    # Current product ownership (Boolean)
    'ind_ahor_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Savings Account'},
    'ind_aval_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Guarantees'},
    'ind_cco_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Checking Account'},
    'ind_cder_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Derivatives'},
    'ind_cno_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Payroll'},
    'ind_ctju_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Junior Account'},
    'ind_ctma_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Particular Account number 3'},
    'ind_ctop_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Particular Account number 1'},
    'ind_ctpp_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Particular Account number 2'},
    'ind_deco_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Short-term Deposit'},
    'ind_deme_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Medium-term Deposit'},
    'ind_dela_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Long-term Deposit'},
    'ind_ecue_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Digital Account'},
    'ind_fond_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Funds'},
    'ind_hip_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Mortgage'},
    'ind_plan_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Pension Plan'},
    'ind_pres_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Loan'},
    'ind_reca_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Tax Account'},
    'ind_tjcr_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Credit Card'},
    'ind_valo_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Securities'},
    'ind_viv_fin_ult1': {'type': 'bool', 'required': True, 'description': 'Home Account'},
    'ind_nomina_ult1': {'type': 'bool', 'required': True, 'description': 'Payroll Account'},
    'ind_nom_pens_ult1': {'type': 'bool', 'required': True, 'description': 'Pensions'},
    'ind_recibo_ult1': {'type': 'bool', 'required': True, 'description': 'Direct Debit'},

    # Lag-3 features (Boolean)
    'ind_ahor_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Savings Account lag3'},
    'ind_aval_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Guarantees lag3'},
    'ind_cco_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Checking Account lag3'},
    'ind_cder_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Derivatives lag3'},
    'ind_cno_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Payroll Account lag3'},
    'ind_ctju_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Junior Account lag3'},
    'ind_ctma_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Particular Account number 3 lag3'},
    'ind_ctop_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Particular Account number 1 lag3'},
    'ind_ctpp_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Particular Account number 2 lag3'},
    'ind_deco_fin_ult1_lag3': {'type': 'bool', 'required': True, 'description': 'Short-term Deposit lag3'},
    'n_products_lag3': {'type': 'int', 'required': True, 'min': 0, 'description': 'Number of Products lag3'},
    
    # Lag-6 features (Boolean)
    'ind_ahor_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Savings Account lag6'},
    'ind_aval_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Guarantees lag6'},
    'ind_cco_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Checking Account lag6'},
    'ind_cder_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Derivatives lag6'},
    'ind_cno_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Payroll Account lag6'},
    'ind_ctju_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Junior Account lag6'},
    'ind_ctma_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Particular Account number 3 lag6'},
    'ind_ctop_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Particular Account number 1 lag6'},
    'ind_ctpp_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Particular Account number 2 lag6'},
    'ind_deco_fin_ult1_lag6': {'type': 'bool', 'required': True, 'description': 'Short-term Deposit lag6'},
    'n_products_lag6': {'type': 'int', 'required': True, 'min': 0, 'description': 'Number of Products lag6'},
    
    # Acquired recently features (Int8: -1, 0, 1)
    'ind_ahor_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Savings Account acquired recently'},
    'ind_aval_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Guarantees acquired recently'},
    'ind_cco_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Checking Account acquired recently'},
    'ind_cder_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Derivatives acquired recently'},
    'ind_cno_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll Account acquired recently'},
    'ind_ctju_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Junior Account acquired recently'},
    'ind_ctma_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Particular Account number 3 acquired recently'},
    'ind_ctop_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Particular Account number 1 acquired recently'},
    'ind_ctpp_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Particular Account number 2 acquired recently'},
    'ind_deco_fin_ult1_acquired_recently': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Short-term Deposit acquired recently'},
    
    # Interaction features (Int8: -1, 0, 1)
    'ind_nomina_ult1_ind_nom_pens_ult1_interaction': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll Account-Pensions interaction'},
    'ind_cno_fin_ult1_ind_nom_pens_ult1_interaction': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll-Pensions interaction'},
    'ind_cno_fin_ult1_ind_nomina_ult1_interaction': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll Account-Payroll interaction'},
    'ind_cno_fin_ult1_ind_recibo_ult1_interaction': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll Account-Direct Debit interaction'},
    'ind_nomina_ult1_ind_recibo_ult1_interaction': {'type': 'int', 'required': True, 'min': -1, 'max': 1, 'description': 'Payroll Account-Direct Debit interaction'},
}

# List of all required feature names (excluding targets)
REQUIRED_FEATURES = [
    'fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 'sexo', 'age',
    'ind_nuevo', 'indrel', 'indrel_1mes', 'tiprel_1mes', 'indresi', 'canal_entrada',
    'indfall', 'cod_prov', 'ind_actividad_cliente', 'renta', 'segmento',
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',
    'customer_period',
    'ind_ahor_fin_ult1_lag3', 'ind_aval_fin_ult1_lag3', 'ind_cco_fin_ult1_lag3',
    'ind_cder_fin_ult1_lag3', 'ind_cno_fin_ult1_lag3', 'ind_ctju_fin_ult1_lag3',
    'ind_ctma_fin_ult1_lag3', 'ind_ctop_fin_ult1_lag3', 'ind_ctpp_fin_ult1_lag3',
    'ind_deco_fin_ult1_lag3', 'n_products_lag3',
    'ind_ahor_fin_ult1_lag6', 'ind_aval_fin_ult1_lag6', 'ind_cco_fin_ult1_lag6',
    'ind_cder_fin_ult1_lag6', 'ind_cno_fin_ult1_lag6', 'ind_ctju_fin_ult1_lag6',
    'ind_ctma_fin_ult1_lag6', 'ind_ctop_fin_ult1_lag6', 'ind_ctpp_fin_ult1_lag6',
    'ind_deco_fin_ult1_lag6', 'n_products_lag6',
    'ind_ahor_fin_ult1_acquired_recently', 'ind_aval_fin_ult1_acquired_recently',
    'ind_cco_fin_ult1_acquired_recently', 'ind_cder_fin_ult1_acquired_recently',
    'ind_cno_fin_ult1_acquired_recently', 'ind_ctju_fin_ult1_acquired_recently',
    'ind_ctma_fin_ult1_acquired_recently', 'ind_ctop_fin_ult1_acquired_recently',
    'ind_ctpp_fin_ult1_acquired_recently', 'ind_deco_fin_ult1_acquired_recently',
    'ind_nomina_ult1_ind_nom_pens_ult1_interaction',
    'ind_cno_fin_ult1_ind_nom_pens_ult1_interaction',
    'ind_cno_fin_ult1_ind_nomina_ult1_interaction',
    'ind_cno_fin_ult1_ind_recibo_ult1_interaction',
    'ind_nomina_ult1_ind_recibo_ult1_interaction',
]

# Product names for recommendations
PRODUCT_NAMES = [
    'target_ahor_fin', 'target_aval_fin', 'target_cco_fin', 'target_cder_fin',
    'target_cno_fin', 'target_ctju_fin', 'target_ctma_fin', 'target_ctop_fin',
    'target_ctpp_fin', 'target_deco_fin', 'target_deme_fin', 'target_dela_fin',
    'target_ecue_fin', 'target_fond_fin', 'target_hip_fin', 'target_plan_fin',
    'target_pres_fin', 'target_reca_fin', 'target_tjcr_fin', 'target_valo_fin',
    'target_viv_fin', 'target_nomina', 'target_nom_pens', 'target_recibo',
]


# ---------- Request Models ---------- #
class CustomerFeatures(BaseModel):
    '''
       Customer features for prediction request.
    '''
    
    # Numeric features
    age: int = Field(..., ge=0, le=150, description='Customer age')
    renta: Optional[float] = Field(None, description='Household income')
    customer_period: int = Field(..., description='Customer tenure period')
    
    # Boolean features
    ind_nuevo: bool = Field(..., description='New customer indicator')
    indresi: bool = Field(..., description='Residence indicator')
    indfall: bool = Field(..., description='Deceased indicator')
    ind_actividad_cliente: bool = Field(..., description='Activity indicator')
    
    # Current product ownership
    ind_ahor_fin_ult1: bool = Field(False, description='Saving Account')
    ind_aval_fin_ult1: bool = Field(False, description='Guarantees')
    ind_cco_fin_ult1: bool = Field(False, description='Current Accounts')
    ind_cder_fin_ult1: bool = Field(False, description='Derivada Account')
    ind_cno_fin_ult1: bool = Field(False, description='Payroll Account')
    ind_ctju_fin_ult1: bool = Field(False, description='Junior Account')
    ind_ctma_fin_ult1: bool = Field(False, description='Más particular Account')
    ind_ctop_fin_ult1: bool = Field(False, description='particular Account')
    ind_ctpp_fin_ult1: bool = Field(False, description='particular Plus Account')
    ind_deco_fin_ult1: bool = Field(False, description='Short-term deposits')
    ind_deme_fin_ult1: bool = Field(False, description='Medium-term deposits')
    ind_dela_fin_ult1: bool = Field(False, description='Long-term deposits')
    ind_ecue_fin_ult1: bool = Field(False, description='e-account')
    ind_fond_fin_ult1: bool = Field(False, description='Funds')
    ind_hip_fin_ult1: bool = Field(False, description='Mortgage')
    ind_plan_fin_ult1: bool = Field(False, description='Pensions')
    ind_pres_fin_ult1: bool = Field(False, description='Loans')
    ind_reca_fin_ult1: bool = Field(False, description='Taxes')
    ind_tjcr_fin_ult1: bool = Field(False, description='Credit Card')
    ind_valo_fin_ult1: bool = Field(False, description='Securities')
    ind_viv_fin_ult1: bool = Field(False, description='Home Account')
    ind_nomina_ult1: bool = Field(False, description='Payroll')
    ind_nom_pens_ult1: bool = Field(False, description='Pensions payroll')
    ind_recibo_ult1: bool = Field(False, description='Direct Debit')
    
    # Lag-3 features
    ind_ahor_fin_ult1_lag3: bool = Field(False)
    ind_aval_fin_ult1_lag3: bool = Field(False)
    ind_cco_fin_ult1_lag3: bool = Field(False)
    ind_cder_fin_ult1_lag3: bool = Field(False)
    ind_cno_fin_ult1_lag3: bool = Field(False)
    ind_ctju_fin_ult1_lag3: bool = Field(False)
    ind_ctma_fin_ult1_lag3: bool = Field(False)
    ind_ctop_fin_ult1_lag3: bool = Field(False)
    ind_ctpp_fin_ult1_lag3: bool = Field(False)
    ind_deco_fin_ult1_lag3: bool = Field(False)
    n_products_lag3: int = Field(0, ge=0, description='Number of products lag3')
    
    # Lag-6 features
    ind_ahor_fin_ult1_lag6: bool = Field(False)
    ind_aval_fin_ult1_lag6: bool = Field(False)
    ind_cco_fin_ult1_lag6: bool = Field(False)
    ind_cder_fin_ult1_lag6: bool = Field(False)
    ind_cno_fin_ult1_lag6: bool = Field(False)
    ind_ctju_fin_ult1_lag6: bool = Field(False)
    ind_ctma_fin_ult1_lag6: bool = Field(False)
    ind_ctop_fin_ult1_lag6: bool = Field(False)
    ind_ctpp_fin_ult1_lag6: bool = Field(False)
    ind_deco_fin_ult1_lag6: bool = Field(False)
    n_products_lag6: int = Field(0, ge=0, description='Number of products lag6')
    
    # Acquired recently features
    ind_ahor_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_aval_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_cco_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_cder_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_cno_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_ctju_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_ctma_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_ctop_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_ctpp_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    ind_deco_fin_ult1_acquired_recently: int = Field(0, ge=-1, le=1)
    
    # Interaction features
    ind_nomina_ult1_ind_nom_pens_ult1_interaction: int = Field(0, ge=-1, le=1)
    ind_cno_fin_ult1_ind_nom_pens_ult1_interaction: int = Field(0, ge=-1, le=1)
    ind_cno_fin_ult1_ind_nomina_ult1_interaction: int = Field(0, ge=-1, le=1)
    ind_cno_fin_ult1_ind_recibo_ult1_interaction: int = Field(0, ge=-1, le=1)
    ind_nomina_ult1_ind_recibo_ult1_interaction: int = Field(0, ge=-1, le=1)
    
    class Config:
        extra = 'allow'  # Allow extra fields to be passed through


class PredictionRequest(BaseModel):
    '''
        Request model for prediction endpoint.
    '''
    
    customer_id: str = Field(..., description='Unique customer identifier')
    features: CustomerFeatures = Field(..., description='Customer features for prediction')
    top_k: int = Field(7, ge=1, le=24, description='Number of top recommendations to return')
    
    class Config:
        json_schema_extra = {
            'example': {
                'customer_id': '12345678',
                'features': {
                    'fecha_dato': '2016-05-28',
                    'ncodpers': '12345678',
                    'age': 35,
                    'customer_period': 12,
                    'ind_nuevo': False,
                    'indresi': True,
                    'indfall': False,
                    'ind_actividad_cliente': True,
                    'ind_cco_fin_ult1': True,
                },
                'top_k': 7
            }
        }


class BatchPredictionRequest(BaseModel):
    '''
       Request model for batch prediction endpoint.
    '''
    
    customers: List[PredictionRequest] = Field(..., description='List of customer prediction requests')
    
    class Config:
        json_schema_extra = {
            'example': {
                'customers': [
                    {
                        'customer_id': '12345678',
                        'features': {'ncodpers': '12345678', 'age': 35},
                        'top_k': 7
                    }
                ]
            }
        }


# ---------- Response Models ---------- #
class ProductRecommendation(BaseModel):
    '''
       Single product recommendation.
    '''
    
    product_id: str = Field(..., description='Product identifier')
    product_name: str = Field(..., description='Human-readable product name')
    probability: float = Field(..., ge=0, le=1, description='Recommendation probability')
    rank: int = Field(..., ge=1, description='Recommendation rank')


class PredictionResponse(BaseModel):
    '''
       Response model for prediction endpoint.
    '''
    
    customer_id: str = Field(..., description='Customer identifier')
    recommendations: List[ProductRecommendation] = Field(..., description='Top-K product recommendations')
    latency_ms: float = Field(..., description='Prediction latency in milliseconds')
    model_version: str = Field(..., description='Model version used for prediction')


class BatchPredictionResponse(BaseModel):
    '''
       Response model for batch prediction endpoint.
    '''
    
    predictions: List[PredictionResponse] = Field(..., description='List of prediction responses')
    total_latency_ms: float = Field(..., description='Total batch processing latency in milliseconds')
    batch_size: int = Field(..., description='Number of predictions in batch')


class HealthResponse(BaseModel):
    '''
       Response model for health check endpoint.
    '''
    
    status: str = Field(..., description='Health status')
    model_loaded: bool = Field(..., description='Whether model is loaded')
    version: str = Field(..., description='API version')


class ValidationError(BaseModel):
    '''
        Validation error response.
    '''
    
    field: str = Field(..., description='Field that failed validation')
    message: str = Field(..., description='Error message')
    expected_type: Optional[str] = Field(None, description='Expected data type')


class ErrorResponse(BaseModel):
    '''
       Error response model.
    '''
    
    error: str = Field(..., description='Error type')
    message: str = Field(..., description='Error message')
    details: Optional[List[ValidationError]] = Field(None, description='Validation error details')


# ---------- Exports ---------- #
__all__ = [
    'FEATURE_SCHEMA',
    'REQUIRED_FEATURES', 
    'PRODUCT_NAMES',
    'CustomerFeatures',
    'PredictionRequest',
    'BatchPredictionRequest',
    'ProductRecommendation',
    'PredictionResponse',
    'BatchPredictionResponse',
    'HealthResponse',
    'ValidationError',
    'ErrorResponse',
]

