"""Centralized input validation and sanitization (issue #333)."""
from __future__ import annotations

import re
import html
from typing import Any, Optional
from urllib.parse import urlparse

from pydantic import BaseModel, validator
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError


class ValidationError(Exception):
    """Custom validation error."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        self.message = message
        self.field = field
        super().__init__(message)


class InputValidator:
    """Centralized input validation and sanitization."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|EXEC|UNION)\b)",
        r"(--|;|\/\*|\*\/)",
        r"(\bOR\b.*=.*\bOR\b)",
        r"(\bAND\b.*=.*\bAND\b)",
        r"(\bWHERE\b.*=)",
        r"(\bEXEC\b\(|\bEXECUTE\b\()",
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
        r"eval\(",
        r"fromCharCode",
    ]

    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize a string input."""
        if not isinstance(value, str):
            return value
        
        # Remove null bytes
        value = value.replace("\x00", "")
        
        # Escape HTML entities
        value = html.escape(value)
        
        return value

    @classmethod
    def check_sql_injection(cls, value: str) -> bool:
        """Check if input contains SQL injection patterns."""
        if not isinstance(value, str):
            return False
        
        value_upper = value.upper()
        for pattern in cls.SQL_INJECTION_PATTERNS:
            if re.search(pattern, value_upper, re.IGNORECASE):
                return True
        return False

    @classmethod
    def check_xss(cls, value: str) -> bool:
        """Check if input contains XSS patterns."""
        if not isinstance(value, str):
            return False
        
        for pattern in cls.XSS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        return False

    @classmethod
    def validate_url(cls, value: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(value)
            return all([result.scheme, result.netloc])
        except Exception:  # noqa: BLE001
            return False

    @classmethod
    def validate_email(cls, value: str) -> bool:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return bool(re.match(pattern, value))

    @classmethod
    def validate_public_key(cls, value: str) -> bool:
        """Validate Stellar public key format (56 characters, base32)."""
        if len(value) != 56:
            return False
        pattern = r"^[G][A-Z2-7]{55}$"
        return bool(re.match(pattern, value))

    @classmethod
    def sanitize_input(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Sanitize all string values in a dictionary."""
        sanitized = {}
        for key, value in data.items():
            if isinstance(value, str):
                sanitized[key] = cls.sanitize_string(value)
            elif isinstance(value, dict):
                sanitized[key] = cls.sanitize_input(value)
            elif isinstance(value, list):
                sanitized[key] = [
                    cls.sanitize_string(item) if isinstance(item, str) else item
                    for item in value
                ]
            else:
                sanitized[key] = value
        return sanitized

    @classmethod
    def validate_input(cls, data: dict[str, Any], field_validators: dict[str, list]) -> None:
        """Validate input data against field-specific validators."""
        for field, validators in field_validators.items():
            if field not in data:
                continue
            
            value = data[field]
            
            for validator_func in validators:
                if not validator_func(value):
                    raise ValidationError(
                        f"Invalid value for field '{field}'",
                        field=field,
                    )


class SQLInjectionAuditor:
    """Audit SQL queries for injection vulnerabilities."""

    @classmethod
    def audit_query(cls, query: str) -> dict[str, Any]:
        """Audit a SQL query for potential injection vulnerabilities."""
        issues = []
        
        # Check for string concatenation patterns
        if re.search(r'["\'].*\+.*["\']', query):
            issues.append({
                "severity": "high",
                "message": "String concatenation detected - use parameterized queries",
            })
        
        # Check for direct user input patterns
        if re.search(r'(format|%|\.format)\(', query):
            issues.append({
                "severity": "medium",
                "message": "String formatting detected - use parameterized queries",
            })
        
        # Check for EXEC/EXECUTE with user input
        if re.search(r'(EXEC|EXECUTE)\s*\(', query, re.IGNORECASE):
            issues.append({
                "severity": "high",
                "message": "Dynamic SQL execution detected",
            })
        
        return {
            "safe": len(issues) == 0,
            "issues": issues,
        }


class XSSPreventionMiddleware:
    """Middleware to prevent XSS in API responses."""

    @classmethod
    def sanitize_response(cls, data: Any) -> Any:
        """Sanitize response data to prevent XSS."""
        if isinstance(data, str):
            return cls.sanitize_string(data)
        elif isinstance(data, dict):
            return {key: cls.sanitize_response(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [cls.sanitize_response(item) for item in data]
        else:
            return data

    @classmethod
    def sanitize_string(cls, value: str) -> str:
        """Sanitize a string for output."""
        # Escape HTML entities
        return html.escape(value)


class FileUploadValidator:
    """Validate file uploads for security."""

    ALLOWED_MIME_TYPES = {
        "image/jpeg",
        "image/png",
        "image/gif",
        "application/pdf",
        "text/plain",
        "application/json",
    }

    ALLOWED_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".txt", ".json"
    }

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

    @classmethod
    def validate_file(
        cls,
        filename: str,
        content_type: str,
        file_size: int,
    ) -> dict[str, Any]:
        """Validate a file upload."""
        errors = []

        # Check file extension
        ext = cls._get_extension(filename)
        if ext.lower() not in cls.ALLOWED_EXTENSIONS:
            errors.append(f"File extension '{ext}' is not allowed")

        # Check MIME type
        if content_type not in cls.ALLOWED_MIME_TYPES:
            errors.append(f"MIME type '{content_type}' is not allowed")

        # Check file size
        if file_size > cls.MAX_FILE_SIZE:
            errors.append(f"File size {file_size} exceeds maximum {cls.MAX_FILE_SIZE}")

        # Check for malicious filename patterns
        if cls._has_malicious_filename(filename):
            errors.append("Filename contains malicious patterns")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }

    @classmethod
    def _get_extension(cls, filename: str) -> str:
        """Extract file extension from filename."""
        if "." not in filename:
            return ""
        return filename.rsplit(".", 1)[1]

    @classmethod
    def _has_malicious_filename(cls, filename: str) -> bool:
        """Check for malicious filename patterns."""
        malicious_patterns = [
            "..",  # Directory traversal
            "\x00",  # Null byte
            "/",  # Path separator
            "\\",  # Windows path separator
        ]
        
        for pattern in malicious_patterns:
            if pattern in filename:
                return True
        
        return False


# Common field validators
COMMON_VALIDATORS = {
    "email": [InputValidator.validate_email],
    "url": [InputValidator.validate_url],
    "public_key": [InputValidator.validate_public_key],
}
