"""API input validation using Pydantic for issue #303.

Provides comprehensive schema validation for all API endpoints including:
- Transaction validation
- Account validation
- Fraud detection input validation
- Feature data validation
- Custom validation rules
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from decimal import Decimal

from pydantic import BaseModel, Field, validator, field_validator, ConfigDict
from pydantic import ValidationError as PydanticValidationError


class ValidationError:
    """Structured validation error for API responses."""

    def __init__(
        self,
        field: str,
        message: str,
        error_type: str,
        value: Any = None,
    ):
        self.field = field
        self.message = message
        self.error_type = error_type
        self.value = value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "field": self.field,
            "message": self.message,
            "error_type": self.error_type,
            "value": self.value,
        }


class ValidationResult:
    """Result of validation with errors and status."""

    def __init__(self, is_valid: bool, errors: List[ValidationError] = None):
        self.is_valid = is_valid
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "error_count": len(self.errors),
        }


# ─── Transaction Validation ───────────────────────────────────────────────

class TransactionInput(BaseModel):
    """Schema for transaction input validation."""

    hash: str = Field(..., min_length=64, max_length=64, description="Transaction hash")
    ledger_sequence: int = Field(..., gt=0, description="Ledger sequence number")
    source_account: str = Field(..., min_length=56, max_length=56, description="Source account public key")
    destination_account: Optional[str] = Field(None, min_length=56, max_length=56, description="Destination account public key")
    amount: Optional[float] = Field(None, ge=0, description="Transaction amount")
    asset_code: Optional[str] = Field(None, max_length=12, description="Asset code")
    asset_issuer: Optional[str] = Field(None, min_length=56, max_length=56, description="Asset issuer")
    fee: int = Field(..., ge=0, description="Transaction fee in stroops")
    operation_type: Optional[str] = Field(None, max_length=32, description="Operation type")
    successful: bool = Field(default=True, description="Transaction success status")
    memo_type: Optional[str] = Field(None, max_length=16, description="Memo type")
    memo: Optional[str] = Field(None, max_length=28, description="Memo content")
    created_at: datetime = Field(..., description="Transaction timestamp")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator('hash')
    @classmethod
    def validate_hash(cls, v: str) -> str:
        """Validate transaction hash format (hex string)."""
        if not re.match(r'^[a-fA-F0-9]{64}$', v):
            raise ValueError('Transaction hash must be a 64-character hexadecimal string')
        return v.lower()

    @field_validator('source_account', 'destination_account', 'asset_issuer')
    @classmethod
    def validate_public_key(cls, v: Optional[str]) -> Optional[str]:
        """Validate Stellar public key format."""
        if v is not None:
            if not re.match(r'^G[A-Za-z0-9]{55}$', v):
                raise ValueError('Invalid Stellar public key format')
        return v

    @field_validator('asset_code')
    @classmethod
    def validate_asset_code(cls, v: Optional[str]) -> Optional[str]:
        """Validate asset code format."""
        if v is not None:
            if not re.match(r'^[A-Za-z0-9]{1,12}$', v):
                raise ValueError('Asset code must be 1-12 alphanumeric characters')
        return v.upper() if v else v

    @field_validator('memo_type')
    @classmethod
    def validate_memo_type(cls, v: Optional[str]) -> Optional[str]:
        """Validate memo type against Stellar types."""
        valid_types = ['none', 'text', 'id', 'hash']
        if v is not None and v.lower() not in valid_types:
            raise ValueError(f'Memo type must be one of: {", ".join(valid_types)}')
        return v.lower() if v else v


# ─── Account Validation ───────────────────────────────────────────────────

class AccountInput(BaseModel):
    """Schema for account input validation."""

    account_id: str = Field(..., min_length=56, max_length=56, description="Account public key")
    balance: Optional[float] = Field(None, ge=0, description="Account balance")
    sequence: Optional[int] = Field(None, ge=0, description="Account sequence number")
    home_domain: Optional[str] = Field(None, max_length=253, description="Home domain")
    flags: int = Field(default=0, ge=0, le=255, description="Account flags")
    last_modified_ledger: Optional[int] = Field(None, gt=0, description="Last modified ledger")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator('account_id')
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        """Validate Stellar account ID format."""
        if not re.match(r'^G[A-Za-z0-9]{55}$', v):
            raise ValueError('Invalid Stellar account ID format')
        return v

    @field_validator('home_domain')
    @classmethod
    def validate_home_domain(cls, v: Optional[str]) -> Optional[str]:
        """Validate home domain format."""
        if v is not None:
            # Basic domain validation
            if not re.match(r'^[a-zA-Z0-9][a-zA-Z0-9-]{0,61}[a-zA-Z0-9]?$', v):
                raise ValueError('Invalid home domain format')
        return v.lower() if v else v


# ─── Fraud Detection Validation ───────────────────────────────────────────

class EdgeInput(BaseModel):
    """Schema for graph edge input validation."""

    src: str = Field(..., min_length=56, max_length=56, description="Source account")
    dst: str = Field(..., min_length=56, max_length=56, description="Destination account")
    amount: float = Field(default=0.0, ge=0, description="Transaction amount")
    timestamp: float = Field(default=0.0, ge=0, description="Unix timestamp")
    asset: str = Field(default="XLM", max_length=12, description="Asset code")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator('src', 'dst')
    @classmethod
    def validate_account(cls, v: str) -> str:
        """Validate account format."""
        if not re.match(r'^G[A-Za-z0-9]{55}$', v):
            raise ValueError('Invalid Stellar account format')
        return v

    @field_validator('asset')
    @classmethod
    def validate_asset(cls, v: str) -> str:
        """Validate asset code."""
        if not re.match(r'^[A-Za-z0-9]{1,12}$', v):
            raise ValueError('Invalid asset code format')
        return v.upper()


class ScoreRequestInput(BaseModel):
    """Schema for fraud scoring request validation."""

    accounts: List[str] = Field(..., min_length=1, max_length=50, description="List of accounts to score")
    edges: List[EdgeInput] = Field(default_factory=list, description="Graph edges")

    @field_validator('accounts')
    @classmethod
    def validate_accounts(cls, v: List[str]) -> List[str]:
        """Validate all accounts in the list."""
        for account in v:
            if not re.match(r'^G[A-Za-z0-9]{55}$', account):
                raise ValueError(f'Invalid account format: {account}')
        return list(set(v))  # Remove duplicates


# ─── Feature Data Validation ────────────────────────────────────────────────

class FeatureDataInput(BaseModel):
    """Schema for feature data validation."""

    account_id: str = Field(..., description="Account identifier")
    features: Dict[str, Union[int, float, str, bool]] = Field(..., description="Feature dictionary")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Feature timestamp")

    @field_validator('account_id')
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        """Validate account ID format."""
        if not re.match(r'^[A-Za-z0-9_-]{1,128}$', v):
            raise ValueError('Invalid account ID format')
        return v

    @field_validator('features')
    @classmethod
    def validate_features(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate feature types and names."""
        if not v:
            raise ValueError('Features dictionary cannot be empty')

        for key, value in v.items():
            # Validate feature name
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', key):
                raise ValueError(f'Invalid feature name: {key}')

            # Validate feature value type
            if not isinstance(value, (int, float, str, bool)):
                raise ValueError(f'Feature {key} has invalid type: {type(value).__name__}')

            # Check for NaN or infinite values
            if isinstance(value, float):
                if value != value:  # NaN check
                    raise ValueError(f'Feature {key} contains NaN')
                if abs(value) == float('inf'):
                    raise ValueError(f'Feature {key} contains infinite value')

        return v


# ─── Ledger Validation ───────────────────────────────────────────────────

class LedgerInput(BaseModel):
    """Schema for ledger input validation."""

    sequence: int = Field(..., gt=0, description="Ledger sequence number")
    hash: str = Field(..., min_length=64, max_length=64, description="Ledger hash")
    prev_hash: Optional[str] = Field(None, min_length=64, max_length=64, description="Previous ledger hash")
    closed_at: datetime = Field(..., description="Ledger close time")
    successful_transaction_count: int = Field(default=0, ge=0, description="Number of successful transactions")
    failed_transaction_count: int = Field(default=0, ge=0, description="Number of failed transactions")
    operation_count: int = Field(default=0, ge=0, description="Number of operations")
    total_coins: Optional[float] = Field(None, ge=0, description="Total coins")
    fee_pool: Optional[float] = Field(None, ge=0, description="Fee pool")
    base_fee_in_stroops: Optional[int] = Field(None, ge=0, description="Base fee")
    protocol_version: Optional[int] = Field(None, ge=0, description="Protocol version")

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator('hash', 'prev_hash')
    @classmethod
    def validate_hash(cls, v: Optional[str]) -> Optional[str]:
        """Validate ledger hash format."""
        if v is not None:
            if not re.match(r'^[a-fA-F0-9]{64}$', v):
                raise ValueError('Ledger hash must be a 64-character hexadecimal string')
        return v.lower() if v else v


# ─── Validation Pipeline ─────────────────────────────────────────────────

class ValidationPipeline:
    """Pipeline for validating data through multiple stages."""

    def __init__(self):
        self.validators = []

    def add_validator(self, validator_class, **kwargs):
        """Add a validator to the pipeline."""
        self.validators.append((validator_class, kwargs))
        return self

    def validate(self, data: Dict[str, Any]) -> ValidationResult:
        """Run data through all validators in the pipeline."""
        errors = []

        for validator_class, kwargs in self.validators:
            try:
                validator_class(**kwargs).model_validate(data)
            except PydanticValidationError as e:
                for error in e.errors():
                    field = '.'.join(str(loc) for loc in error['loc'])
                    errors.append(ValidationError(
                        field=field,
                        message=error['msg'],
                        error_type=error['type'],
                        value=error.get('input')
                    ))

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_batch(self, data_list: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Validate a batch of data."""
        return [self.validate(data) for data in data_list]


# ─── Custom Validation Rules ─────────────────────────────────────────────

class CustomValidationRules:
    """Collection of custom validation rules for business logic."""

    @staticmethod
    def validate_transaction_limits(amount: float, max_amount: float = 1_000_000_000) -> ValidationResult:
        """Validate transaction amount against limits."""
        errors = []
        if amount > max_amount:
            errors.append(ValidationError(
                field='amount',
                message=f'Transaction amount {amount} exceeds maximum limit {max_amount}',
                error_type='EXCEEDS_LIMIT',
                value=amount
            ))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_account_age(created_at: datetime, min_age_days: int = 0) -> ValidationResult:
        """Validate account age."""
        errors = []
        age_days = (datetime.utcnow() - created_at).days
        if age_days < min_age_days:
            errors.append(ValidationError(
                field='created_at',
                message=f'Account age {age_days} days is below minimum {min_age_days} days',
                error_type='INSUFFICIENT_AGE',
                value=age_days
            ))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_feature_range(features: Dict[str, float], ranges: Dict[str, tuple]) -> ValidationResult:
        """Validate feature values against expected ranges."""
        errors = []
        for feature, value in features.items():
            if feature in ranges:
                min_val, max_val = ranges[feature]
                if not (min_val <= value <= max_val):
                    errors.append(ValidationError(
                        field=feature,
                        message=f'Feature {feature} value {value} outside range [{min_val}, {max_val}]',
                        error_type='OUT_OF_RANGE',
                        value=value
                    ))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    @staticmethod
    def validate_no_negative_features(features: Dict[str, Union[int, float]]) -> ValidationResult:
        """Ensure no features have negative values (where applicable)."""
        errors = []
        for feature, value in features.items():
            if isinstance(value, (int, float)) and value < 0:
                errors.append(ValidationError(
                    field=feature,
                    message=f'Feature {feature} has negative value {value}',
                    error_type='NEGATIVE_VALUE',
                    value=value
                ))
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


# ─── Convenience Functions ───────────────────────────────────────────────

def validate_transaction_input(data: Dict[str, Any]) -> ValidationResult:
    """Validate transaction input using Pydantic schema."""
    try:
        TransactionInput(**data)
        return ValidationResult(is_valid=True)
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            errors.append(ValidationError(
                field=field,
                message=error['msg'],
                error_type=error['type'],
                value=error.get('input')
            ))
        return ValidationResult(is_valid=False, errors=errors)


def validate_account_input(data: Dict[str, Any]) -> ValidationResult:
    """Validate account input using Pydantic schema."""
    try:
        AccountInput(**data)
        return ValidationResult(is_valid=True)
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            errors.append(ValidationError(
                field=field,
                message=error['msg'],
                error_type=error['type'],
                value=error.get('input')
            ))
        return ValidationResult(is_valid=False, errors=errors)


def validate_score_request(data: Dict[str, Any]) -> ValidationResult:
    """Validate fraud scoring request using Pydantic schema."""
    try:
        ScoreRequestInput(**data)
        return ValidationResult(is_valid=True)
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            errors.append(ValidationError(
                field=field,
                message=error['msg'],
                error_type=error['type'],
                value=error.get('input')
            ))
        return ValidationResult(is_valid=False, errors=errors)


def validate_feature_data(data: Dict[str, Any]) -> ValidationResult:
    """Validate feature data using Pydantic schema."""
    try:
        FeatureDataInput(**data)
        return ValidationResult(is_valid=True)
    except PydanticValidationError as e:
        errors = []
        for error in e.errors():
            field = '.'.join(str(loc) for loc in error['loc'])
            errors.append(ValidationError(
                field=field,
                message=error['msg'],
                error_type=error['type'],
                value=error.get('input')
            ))
        return ValidationResult(is_valid=False, errors=errors)
