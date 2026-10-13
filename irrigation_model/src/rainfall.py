"""
Rainfall Processing Module - Handles rainfall input validation and preservation.

This module provides:
- RainfallInput for validated rainfall data
- RainfallResult for structured output
- No automatic effective rainfall calculation
"""
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum
import math


class RainfallStatus(Enum):
    """Status of rainfall processing."""
    VALID = "VALID"
    MISSING = "MISSING"
    INVALID = "INVALID"


class EffectiveRainfallStatus(Enum):
    """Status of effective rainfall calculation."""
    NOT_CALCULATED = "NOT_CALCULATED"
    CALCULATED = "CALCULATED"
    NOT_AVAILABLE = "NOT_AVAILABLE"


@dataclass
class RainfallInput:
    """
    Validated rainfall input.
    
    Rainfall must be provided in mm.
    Missing optional value is None.
    """
    rainfall_mm: Optional[float] = None
    
    # Validation results
    status: RainfallStatus = RainfallStatus.VALID
    invalid_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate rainfall after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate rainfall input."""
        self.warnings = []
        self.invalid_reason = None
        
        if self.rainfall_mm is None:
            self.status = RainfallStatus.MISSING
            return
        
        # Check if numeric
        if not isinstance(self.rainfall_mm, (int, float)):
            self.status = RainfallStatus.INVALID
            self.invalid_reason = f"rainfall_mm must be numeric, got {type(self.rainfall_mm).__name__}"
            return
        
        # Check for NaN
        if isinstance(self.rainfall_mm, float) and math.isnan(self.rainfall_mm):
            self.status = RainfallStatus.INVALID
            self.invalid_reason = "rainfall_mm is NaN"
            return
        
        # Check for infinity
        if isinstance(self.rainfall_mm, float) and math.isinf(self.rainfall_mm):
            self.status = RainfallStatus.INVALID
            self.invalid_reason = "rainfall_mm is infinite"
            return
        
        # Check for negative
        if self.rainfall_mm < 0:
            self.status = RainfallStatus.INVALID
            self.invalid_reason = f"rainfall_mm cannot be negative: {self.rainfall_mm}"
            return
        
        # Valid
        self.status = RainfallStatus.VALID
    
    def is_valid(self) -> bool:
        """Check if rainfall input is valid."""
        return self.status == RainfallStatus.VALID
    
    def is_missing(self) -> bool:
        """Check if rainfall is missing."""
        return self.status == RainfallStatus.MISSING
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rainfall_mm": self.rainfall_mm,
            "status": self.status.value,
            "invalid_reason": self.invalid_reason,
            "warnings": self.warnings,
        }


@dataclass
class RainfallResult:
    """
    Structured rainfall result.
    
    Preserves rainfall exactly as measured.
    Does NOT automatically calculate effective rainfall.
    """
    # Measured rainfall
    rainfall_mm: Optional[float] = None
    
    # Effective rainfall - NOT automatically calculated
    effective_rainfall_mm: Optional[float] = None
    effective_rainfall_status: EffectiveRainfallStatus = EffectiveRainfallStatus.NOT_CALCULATED
    effective_rainfall_method: Optional[str] = None
    
    # Status
    status: RainfallStatus = RainfallStatus.VALID
    invalid_reason: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    
    # Units
    units: str = "mm"
    
    def is_valid(self) -> bool:
        """Check if rainfall result is valid."""
        return self.status == RainfallStatus.VALID
    
    def has_rainfall(self) -> bool:
        """Check if rainfall was measured (including zero)."""
        return self.rainfall_mm is not None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "rainfall_mm": self.rainfall_mm,
            "effective_rainfall_mm": self.effective_rainfall_mm,
            "effective_rainfall_status": self.effective_rainfall_status.value,
            "effective_rainfall_method": self.effective_rainfall_method,
            "status": self.status.value,
            "invalid_reason": self.invalid_reason,
            "warnings": self.warnings,
            "units": self.units,
        }


def process_rainfall(rainfall_mm: Optional[float]) -> RainfallResult:
    """
    Process rainfall input and return structured result.
    
    Args:
        rainfall_mm: Rainfall in mm (can be None for missing)
    
    Returns:
        RainfallResult with validation and metadata
    """
    input_obj = RainfallInput(rainfall_mm=rainfall_mm)
    
    result = RainfallResult(
        rainfall_mm=rainfall_mm,
        status=input_obj.status,
        invalid_reason=input_obj.invalid_reason,
        warnings=input_obj.warnings,
    )
    
    # Effective rainfall is NOT automatically calculated
    result.effective_rainfall_mm = None
    result.effective_rainfall_status = EffectiveRainfallStatus.NOT_CALCULATED
    result.effective_rainfall_method = None
    
    if input_obj.status == RainfallStatus.INVALID:
        result.warnings.append(f"Invalid rainfall input: {input_obj.invalid_reason}")
    elif input_obj.status == RainfallStatus.MISSING:
        result.warnings.append("Rainfall data not available")
    
    return result


def process_rainfall_from_weather_input(weather_input) -> RainfallResult:
    """
    Process rainfall from a WeatherInput object.
    
    Args:
        weather_input: WeatherInput object with rainfall_mm field
    
    Returns:
        RainfallResult
    """
    return process_rainfall(weather_input.rainfall_mm)