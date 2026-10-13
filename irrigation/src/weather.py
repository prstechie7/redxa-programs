"""
Weather Input Module - Handles weather sensor data validation and ET₀ calculation preparation.

This module provides:
- WeatherInput dataclass for validated weather sensor data
- Input validation for meteorological parameters
- Status tracking for missing/invalid inputs
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum


class WeatherInputStatus(Enum):
    """Status of weather input validation."""
    VALID = "VALID"
    INVALID = "INVALID"
    MISSING_REQUIRED = "MISSING_REQUIRED"
    INSUFFICIENT_FOR_PENMAN_MONTEITH = "INSUFFICIENT_FOR_PENMAN_MONTEITH"
    INSUFFICIENT_FOR_HARGREAVES = "INSUFFICIENT_FOR_HARGREAVES"
    INSUFFICIENT_FOR_ANY = "INSUFFICIENT_FOR_ANY"


class ET0Method(Enum):
    """ET₀ calculation methods."""
    PENMAN_MONTEITH = "penman_monteith"
    HARGREAVES = "hargreaves"
    NOT_CALCULATED = "not_calculated"


@dataclass
class WeatherInput:
    """
    Validated weather sensor input.
    
    All values must be provided in the specified units.
    Missing optional values are None.
    """
    # Required for any ET₀ calculation
    timestamp: datetime
    
    # Temperature (Celsius)
    air_temperature: Optional[float] = None          # Mean air temperature (°C)
    minimum_temperature: Optional[float] = None      # Minimum air temperature (°C)
    maximum_temperature: Optional[float] = None      # Maximum air temperature (°C)
    
    # Humidity (percent, 0-100)
    relative_humidity: Optional[float] = None        # Mean relative humidity (%)
    
    # Precipitation (mm)
    rainfall_mm: Optional[float] = None              # Total rainfall (mm)
    
    # Wind (m/s)
    wind_speed: Optional[float] = None               # Wind speed at 2m height (m/s)
    
    # Radiation (MJ/m²/day)
    solar_radiation: Optional[float] = None          # Solar radiation (MJ/m²/day)
    
    # Location
    latitude: Optional[float] = None                 # Latitude (degrees, positive north)
    elevation: Optional[float] = None                # Elevation (meters above sea level)
    
    # Validation results
    status: WeatherInputStatus = WeatherInputStatus.VALID
    missing_inputs: List[str] = field(default_factory=list)
    invalid_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate all provided weather inputs."""
        self.missing_inputs = []
        self.invalid_inputs = []
        self.warnings = []
        
        # Validate temperature
        if self.air_temperature is not None:
            if not (-50 <= self.air_temperature <= 60):
                self.invalid_inputs.append(f"air_temperature: {self.air_temperature} °C outside valid range [-50, 60]")
        else:
            self.missing_inputs.append("air_temperature")
        
        if self.minimum_temperature is not None:
            if not (-50 <= self.minimum_temperature <= 60):
                self.invalid_inputs.append(f"minimum_temperature: {self.minimum_temperature} °C outside valid range [-50, 60]")
        else:
            self.missing_inputs.append("minimum_temperature")
        
        if self.maximum_temperature is not None:
            if not (-50 <= self.maximum_temperature <= 60):
                self.invalid_inputs.append(f"maximum_temperature: {self.maximum_temperature} °C outside valid range [-50, 60]")
        else:
            self.missing_inputs.append("maximum_temperature")
        
        # Cross-validate temperatures
        if (self.minimum_temperature is not None and 
            self.maximum_temperature is not None and
            self.minimum_temperature > self.maximum_temperature):
            self.invalid_inputs.append("minimum_temperature exceeds maximum_temperature")
        
        if (self.air_temperature is not None and 
            self.minimum_temperature is not None and
            self.maximum_temperature is not None):
            if not (self.minimum_temperature <= self.air_temperature <= self.maximum_temperature):
                self.warnings.append("air_temperature not within [minimum_temperature, maximum_temperature]")
        
        # Validate relative humidity
        if self.relative_humidity is not None:
            if not (0 <= self.relative_humidity <= 100):
                self.invalid_inputs.append(f"relative_humidity: {self.relative_humidity} % outside valid range [0, 100]")
        else:
            self.missing_inputs.append("relative_humidity")
        
        # Validate rainfall
        if self.rainfall_mm is not None:
            if self.rainfall_mm < 0:
                self.invalid_inputs.append(f"rainfall_mm: {self.rainfall_mm} mm cannot be negative")
        else:
            self.missing_inputs.append("rainfall_mm")
        
        # Validate wind speed
        if self.wind_speed is not None:
            if self.wind_speed < 0:
                self.invalid_inputs.append(f"wind_speed: {self.wind_speed} m/s cannot be negative")
        else:
            self.missing_inputs.append("wind_speed")
        
        # Validate solar radiation
        if self.solar_radiation is not None:
            if self.solar_radiation < 0:
                self.invalid_inputs.append(f"solar_radiation: {self.solar_radiation} MJ/m²/day cannot be negative")
        else:
            self.missing_inputs.append("solar_radiation")
        
        # Validate latitude
        if self.latitude is not None:
            if not (-90 <= self.latitude <= 90):
                self.invalid_inputs.append(f"latitude: {self.latitude} degrees outside valid range [-90, 90]")
        else:
            self.missing_inputs.append("latitude")
        
        # Validate elevation
        if self.elevation is not None:
            if self.elevation < -500:
                self.invalid_inputs.append(f"elevation: {self.elevation} m below valid minimum (-500 m)")
        else:
            self.missing_inputs.append("elevation")
        
        # Set overall status based on available methods
        if self.invalid_inputs:
            self.status = WeatherInputStatus.INVALID
        elif self._has_minimum_for_penman_monteith():
            self.status = WeatherInputStatus.VALID
        elif self._has_minimum_for_hargreaves():
            self.status = WeatherInputStatus.INSUFFICIENT_FOR_PENMAN_MONTEITH
        else:
            self.status = WeatherInputStatus.INSUFFICIENT_FOR_ANY
    
    def _has_minimum_for_penman_monteith(self) -> bool:
        """Check if minimum inputs for Penman-Monteith are available."""
        return (
            self.air_temperature is not None and
            self.relative_humidity is not None and
            self.wind_speed is not None and
            self.solar_radiation is not None and
            self.latitude is not None and
            self.elevation is not None
        )
    
    def _has_minimum_for_hargreaves(self) -> bool:
        """Check if minimum inputs for Hargreaves are available."""
        return (
            self.minimum_temperature is not None and
            self.maximum_temperature is not None and
            self.latitude is not None and
            self.timestamp is not None
        )
    
    def can_calculate_penman_monteith(self) -> bool:
        """Check if Penman-Monteith ET₀ can be calculated."""
        return self._has_minimum_for_penman_monteith() and not self.invalid_inputs
    
    def can_calculate_hargreaves(self) -> bool:
        """Check if Hargreaves ET₀ can be calculated."""
        return self._has_minimum_for_hargreaves() and not self.invalid_inputs
    
    def get_missing_for_penman_monteith(self) -> List[str]:
        """Get list of missing inputs for Penman-Monteith."""
        required = ["air_temperature", "relative_humidity", "wind_speed", 
                    "solar_radiation", "latitude", "elevation"]
        return [r for r in required if getattr(self, r) is None]
    
    def get_missing_for_hargreaves(self) -> List[str]:
        """Get list of missing inputs for Hargreaves."""
        required = ["minimum_temperature", "maximum_temperature", "latitude"]
        return [r for r in required if getattr(self, r) is None]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "air_temperature": self.air_temperature,
            "minimum_temperature": self.minimum_temperature,
            "maximum_temperature": self.maximum_temperature,
            "relative_humidity": self.relative_humidity,
            "rainfall_mm": self.rainfall_mm,
            "wind_speed": self.wind_speed,
            "solar_radiation": self.solar_radiation,
            "latitude": self.latitude,
            "elevation": self.elevation,
            "status": self.status.value,
            "missing_inputs": self.missing_inputs,
            "invalid_inputs": self.invalid_inputs,
            "warnings": self.warnings,
        }


@dataclass
class WeatherResult:
    """
    Complete weather/ET₀ calculation result.
    
    Contains all inputs, ET₀ calculation results, ETc, and metadata.
    """
    # Input data
    weather_input: WeatherInput
    
    # ET₀ results
    et0_mm: Optional[float] = None              # Reference evapotranspiration (mm/day)
    et0_method: ET0Method = ET0Method.NOT_CALCULATED
    et0_status: str = "not_calculated"
    
    # Crop coefficient
    kc_value: Optional[float] = None            # Crop coefficient
    kc_stage: Optional[str] = None              # Growth stage
    kc_source: Optional[str] = None             # Source of Kc value
    kc_verification: Optional[str] = None       # Verification status
    
    # ETc results
    etc_mm: Optional[float] = None              # Crop evapotranspiration (mm/day)
    
    # Rainfall
    rainfall_mm: Optional[float] = None         # Total rainfall (mm)
    effective_rainfall_mm: Optional[float] = None  # Effective rainfall (mm) - NOT_CALCULATED by default
    
    # Metadata
    missing_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "weather_input": self.weather_input.to_dict(),
            "et0_mm": self.et0_mm,
            "et0_method": self.et0_method.value,
            "et0_status": self.et0_status,
            "kc_value": self.kc_value,
            "kc_stage": self.kc_stage,
            "kc_source": self.kc_source,
            "kc_verification": self.kc_verification,
            "etc_mm": self.etc_mm,
            "rainfall_mm": self.rainfall_mm,
            "effective_rainfall_mm": self.effective_rainfall_mm,
            "missing_inputs": self.missing_inputs,
            "warnings": self.warnings,
        }