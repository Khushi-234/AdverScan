"""
Custom exceptions for the Model Ingestion module.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union


class ErrorCategory(str, Enum):
    """Typed failure classification for M1 model ingestion."""
    VERSION_MISMATCH = "VERSION_MISMATCH"
    MISSING_DEPENDENCY = "MISSING_DEPENDENCY"
    CONSENT = "CONSENT"
    MALFORMED = "MALFORMED"
    METADATA_INVALID = "METADATA_INVALID"
    ARCH_UNKNOWN = "ARCH_UNKNOWN"
    OTHER = "OTHER"


def classify_exception(exc_or_msg: Union[BaseException, str]) -> ErrorCategory:
    """Classify an exception or error message into a typed ErrorCategory."""
    if isinstance(exc_or_msg, UnsafeModelError):
        return ErrorCategory.CONSENT
    if isinstance(exc_or_msg, (MissingDependencyError, ImportError, ModuleNotFoundError)):
        msg = str(exc_or_msg).lower()
        if "no module named" in msg and any(
            k in msg for k in ("sklearn", "numpy._core", "scipy", "pandas", "copy_reg")
        ):
            return ErrorCategory.VERSION_MISMATCH
        return ErrorCategory.MISSING_DEPENDENCY

    existing_cat = getattr(exc_or_msg, "category", None)
    if isinstance(existing_cat, ErrorCategory) and existing_cat != ErrorCategory.OTHER:
        return existing_cat

    msg = str(exc_or_msg).lower()
    if any(
        k in msg
        for k in (
            "incompatible dtype",
            "can't get attribute",
            "cannot get attribute",
            "node array from the pickle",
            "unpicklingstackunderflow",
            "was serialized with",
            "version mismatch",
        )
    ):
        return ErrorCategory.VERSION_MISMATCH

    if any(
        k in msg
        for k in (
            "explicit user consent",
            "unsafe_allow_pickle",
            "trust_remote_code",
            "custom code",
            "consent",
        )
    ):
        return ErrorCategory.CONSENT

    if any(
        k in msg
        for k in (
            "no module named",
            "is not installed",
            "requires '",
            "requires the '",
            "pip install",
        )
    ):
        return ErrorCategory.MISSING_DEPENDENCY

    if any(
        k in msg
        for k in (
            "state_dict",
            "architecture",
            "model_class",
            "unrecognized model",
            "could not determine model",
        )
    ):
        return ErrorCategory.ARCH_UNKNOWN

    if any(
        k in msg
        for k in (
            "id2label",
            "label2id",
            "validation error for field",
            "config.json",
            "invalid hugging face config",
            "metadata",
        )
    ):
        return ErrorCategory.METADATA_INVALID

    if any(
        k in msg
        for k in (
            "corrupt",
            "malformed",
            "invalid load key",
            "truncated",
            "not a zip file",
            "unable to synchronously open file",
            "magic number",
            "jsondecodeerror",
        )
    ):
        return ErrorCategory.MALFORMED

    return ErrorCategory.OTHER


class ModelIngestionError(Exception):
    """Base exception for model ingestion errors."""

    def __init__(
        self,
        message: str = "",
        *,
        category: Optional[ErrorCategory] = None,
        rejection_reason: Optional[str] = None,
        attempts: Optional[List[Dict[str, Any]]] = None,
    ):
        super().__init__(message)
        self.category: ErrorCategory = category or classify_exception(message)
        self.rejection_reason: Optional[str] = rejection_reason
        self.attempts: List[Dict[str, Any]] = list(attempts) if attempts else []


class ModelLoadError(ModelIngestionError):
    """Raised when loading a model file or model instance fails."""
    pass


class UnsafeModelError(ModelLoadError):
    """Raised when an unsafe deserialization operation is attempted without explicit user consent."""

    def __init__(self, message: str = "", **kwargs: Any):
        kwargs.setdefault("category", ErrorCategory.CONSENT)
        kwargs.setdefault("rejection_reason", "R4")
        super().__init__(message, **kwargs)


class ModelValidationError(ModelIngestionError):
    """Raised when model validation or sanity checking fails."""
    pass


class ConfigurationError(ModelValidationError):
    """Raised when model configuration is missing required fields or invalid."""
    pass


class UnsupportedDomainError(ModelValidationError):
    """Raised when an unsupported data domain is provided."""
    pass


class UnsupportedModelError(ModelIngestionError):
    """Raised when an unsupported model framework or format is provided."""
    pass


class UnsupportedTaskError(ModelIngestionError):
    """Raised when an unsupported task (e.g. forecasting or regression) is provided."""

    def __init__(self, message: str = "", **kwargs: Any):
        kwargs.setdefault("rejection_reason", "R5")
        super().__init__(message, **kwargs)


class MissingDependencyError(ModelIngestionError):
    """Raised when an optional third-party runtime dependency is required but not installed."""

    def __init__(self, message: str = "", **kwargs: Any):
        kwargs.setdefault("category", ErrorCategory.MISSING_DEPENDENCY)
        kwargs.setdefault("rejection_reason", "R6")
        super().__init__(message, **kwargs)


class DatasetFormatError(ModelValidationError):
    """Raised when a dataset format or structure is invalid or unsupported."""
    pass

