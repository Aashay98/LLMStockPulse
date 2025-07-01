"""Custom exceptions for the Stock Insight App."""

class StockAppException(Exception):
    """Base exception for Stock App."""
    pass

class APIException(StockAppException):
    """Exception raised for API-related errors."""
    pass

class DataProcessingException(StockAppException):
    """Exception raised for data processing errors."""
    pass

class ValidationException(StockAppException):
    """Exception raised for validation errors."""
    pass