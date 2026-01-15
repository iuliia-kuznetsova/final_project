'''
    Query Validator

    This module provides functionality to validate API request parameters.
    Validates that all required features exist and have correct types.

    Usage:
    from src.api.validate_query import QueryValidator
    validator = QueryValidator()
    is_valid, errors = validator.validate(request_data)
'''

# ---------- Imports ---------- #
import os
from datetime import date, datetime
from typing import Dict, Any, List, Tuple, Optional, Union
from dotenv import load_dotenv

from src.logging_setup import setup_logging
from src.api.schemas import FEATURE_SCHEMA, REQUIRED_FEATURES


# ---------- Logging setup ---------- #
logger = setup_logging('query_validator')


# ---------- Config ---------- #
config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'config')
load_dotenv(os.path.join(config_dir, '.env'))


# ---------- Type Mapping ---------- #
TYPE_MAPPING = {
    'str': (str,),
    'int': (int,),
    'float': (int, float),
    'bool': (bool, int),  # Allow int 0/1 for bool
    'date': (str, date, datetime),  # Accept string dates
}


# ---------- Query Validator ---------- #
class QueryValidator:
    '''
        Query Validator class.
        This class provides functionality to validate API request data.
        
        Features:
        - Validates presence of all required features
        - Validates data types for each feature
        - Validates numeric ranges where applicable
        - Returns detailed error messages for debugging
    '''

    def __init__(
        self,
        feature_schema: Dict[str, Dict] = None,
        required_features: List[str] = None,
        strict_mode: bool = False
    ):
        '''
            Initialize the validator.
        '''
        self.feature_schema = feature_schema or FEATURE_SCHEMA
        self.required_features = required_features or REQUIRED_FEATURES
        self.strict_mode = strict_mode
        
        logger.info(f'QueryValidator initialized with {len(self.feature_schema)} features, strict_mode={strict_mode}')

    def _validate_type(
        self, 
        value: Any, 
        expected_type: str, 
        field_name: str
    ) -> Tuple[bool, Optional[str]]:
        '''
            Validate that a value matches the expected type.
        '''
        # Handle None values
        if value is None:
            return True, None  # None is allowed, will be handled as missing if required
        
        # Get allowed types
        allowed_types = TYPE_MAPPING.get(expected_type)
        if not allowed_types:
            return False, f"Unknown type '{expected_type}' for field '{field_name}'"
        
        # Special handling for boolean
        if expected_type == 'bool':
            if isinstance(value, bool):
                return True, None
            if isinstance(value, int) and value in (0, 1):
                return True, None
            if isinstance(value, str) and value.lower() in ('true', 'false', '0', '1'):
                return True, None
            return False, f"Field '{field_name}' must be boolean, got {type(value).__name__}: {value}"
        
        # Special handling for date
        if expected_type == 'date':
            if isinstance(value, (date, datetime)):
                return True, None
            if isinstance(value, str):
                try:
                    datetime.strptime(value, '%Y-%m-%d')
                    return True, None
                except ValueError:
                    return False, f"Field '{field_name}' must be date in YYYY-MM-DD format, got: {value}"
            return False, f"Field '{field_name}' must be date, got {type(value).__name__}"
        
        # Standard type check
        if not isinstance(value, allowed_types):
            return False, f"Field '{field_name}' must be {expected_type}, got {type(value).__name__}: {value}"
        
        return True, None

    def _validate_range(
        self, 
        value: Any, 
        schema: Dict, 
        field_name: str
    ) -> Tuple[bool, Optional[str]]:
        '''
            Validate that a numeric value is within the expected range.
        '''
        if value is None:
            return True, None
        
        min_val = schema.get('min')
        max_val = schema.get('max')
        
        if min_val is not None and value < min_val:
            return False, f"Field '{field_name}' must be >= {min_val}, got {value}"
        
        if max_val is not None and value > max_val:
            return False, f"Field '{field_name}' must be <= {max_val}, got {value}"
        
        return True, None

    def check_required_features(
        self, 
        data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        '''
            Check that all required features are present in the data.
        '''
        missing = []
        for feature in self.required_features:
            if feature not in data:
                missing.append(feature)
        
        if missing:
            logger.warning(f"Missing required features: {missing}")
            return False, missing
        
        logger.debug(f"All {len(self.required_features)} required features present")
        return True, []

    def check_feature_types(
        self, 
        data: Dict[str, Any]
    ) -> Tuple[bool, List[Dict[str, str]]]:
        '''
            Validate data types for all features.
        '''
        errors = []
        
        for field_name, value in data.items():
            # Skip unknown fields if not in strict mode
            if field_name not in self.feature_schema:
                if self.strict_mode:
                    errors.append({
                        'field': field_name,
                        'message': f"Unknown field '{field_name}'",
                        'expected_type': None
                    })
                continue
            
            schema = self.feature_schema[field_name]
            expected_type = schema.get('type', 'str')
            
            # Validate type
            is_valid, error_msg = self._validate_type(value, expected_type, field_name)
            if not is_valid:
                errors.append({
                    'field': field_name,
                    'message': error_msg,
                    'expected_type': expected_type
                })
                continue
            
            # Validate range for numeric types
            if expected_type in ('int', 'float') and value is not None:
                is_valid, error_msg = self._validate_range(value, schema, field_name)
                if not is_valid:
                    errors.append({
                        'field': field_name,
                        'message': error_msg,
                        'expected_type': expected_type
                    })
        
        if errors:
            logger.warning(f"Type validation errors: {len(errors)} errors")
            for err in errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {err['field']}: {err['message']}")
        else:
            logger.debug("All feature types validated successfully")
        
        return len(errors) == 0, errors

    def validate_customer_id(
        self, 
        customer_id: Any
    ) -> Tuple[bool, Optional[str]]:
        '''
            Validate customer ID format.
        '''
        if customer_id is None:
            return False, "customer_id is required"
        
        if not isinstance(customer_id, (str, int)):
            return False, f"customer_id must be string or int, got {type(customer_id).__name__}"
        
        # Convert to string for validation
        customer_id_str = str(customer_id)
        
        if len(customer_id_str) == 0:
            return False, "customer_id cannot be empty"
        
        if len(customer_id_str) > 20:
            return False, f"customer_id too long (max 20 chars), got {len(customer_id_str)}"
        
        return True, None

    def validate(
        self, 
        request_data: Dict[str, Any]
    ) -> Tuple[bool, List[Dict[str, str]]]:
        '''
            Validate complete prediction request.
        '''
        errors = []
        
        # Validate customer_id
        customer_id = request_data.get('customer_id')
        is_valid, error_msg = self.validate_customer_id(customer_id)
        if not is_valid:
            errors.append({
                'field': 'customer_id',
                'message': error_msg,
                'expected_type': 'str'
            })
        
        # Get features
        features = request_data.get('features', {})
        if not features:
            errors.append({
                'field': 'features',
                'message': 'features object is required',
                'expected_type': 'object'
            })
            return False, errors
        
        # Convert Pydantic model to dict if necessary
        if hasattr(features, 'model_dump'):
            features = features.model_dump()
        elif hasattr(features, 'dict'):
            features = features.dict()
        
        # Check required features
        has_required, missing = self.check_required_features(features)
        if not has_required:
            for field in missing:
                errors.append({
                    'field': field,
                    'message': f"Required field '{field}' is missing",
                    'expected_type': self.feature_schema.get(field, {}).get('type', 'unknown')
                })
        
        # Check feature types
        types_valid, type_errors = self.check_feature_types(features)
        errors.extend(type_errors)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info(f"Request validation successful for customer_id: {customer_id}")
        else:
            logger.warning(f"Request validation failed with {len(errors)} errors")
        
        return is_valid, errors

    def validate_batch(
        self, 
        requests: List[Dict[str, Any]]
    ) -> Tuple[bool, Dict[int, List[Dict[str, str]]]]:
        '''
            Validate a batch of prediction requests.
        '''
        all_errors = {}
        all_valid = True
        
        for idx, request in enumerate(requests):
            is_valid, errors = self.validate(request)
            if not is_valid:
                all_valid = False
                all_errors[idx] = errors
        
        logger.info(f"Batch validation: {len(requests)} requests, {len(all_errors)} with errors")
        
        return all_valid, all_errors


# ---------- Main ---------- #
if __name__ == '__main__':
    # Test the validator
    validator = QueryValidator()
    
    # Valid test request
    test_request = {
        'customer_id': '12345678',
        'features': {
            'fecha_dato': '2016-05-28',
            'ncodpers': '12345678',
            'ind_empleado': 'A',
            'pais_residencia': 'ES',
            'sexo': 'H',
            'age': 35,
            'ind_nuevo': False,
            'indrel': '1',
            'indrel_1mes': '1',
            'tiprel_1mes': 'A',
            'indresi': True,
            'canal_entrada': 'KHE',
            'indfall': False,
            'cod_prov': '28',
            'ind_actividad_cliente': True,
            'renta': 50000.0,
            'segmento': '02 - PARTICULARES',
            'customer_period': 12,
            # Add all other required features with default values
            **{f: False for f in REQUIRED_FEATURES if f not in [
                'fecha_dato', 'ncodpers', 'ind_empleado', 'pais_residencia', 
                'sexo', 'age', 'ind_nuevo', 'indrel', 'indrel_1mes', 'tiprel_1mes',
                'indresi', 'canal_entrada', 'indfall', 'cod_prov', 
                'ind_actividad_cliente', 'renta', 'segmento', 'customer_period',
                'n_products_lag3', 'n_products_lag6'
            ] + [f for f in REQUIRED_FEATURES if 'acquired_recently' in f or 'interaction' in f]},
            **{f: 0 for f in REQUIRED_FEATURES if 'acquired_recently' in f or 'interaction' in f},
            'n_products_lag3': 0,
            'n_products_lag6': 0,
        }
    }
    
    is_valid, errors = validator.validate(test_request)
    print(f"Validation result: valid={is_valid}")
    if errors:
        print(f"Errors: {errors}")
