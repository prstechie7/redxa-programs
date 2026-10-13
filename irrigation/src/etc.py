"""
Crop Evapotranspiration (ETc) Calculation Module.

Calculates crop evapotranspiration using ET₀ × Kc.
Uses existing Kc values from the threshold database.
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from .et0 import ET0Result, ET0Status
from .threshold_config import get_threshold_config


class ETcStatus(Enum):
    """Status of ETc calculation."""
    CALCULATED = "CALCULATED"
    INVALID_ET0 = "INVALID_ET0"
    Kc_NOT_AVAILABLE = "Kc_NOT_AVAILABLE"
    INVALID_KC = "INVALID_KC"
    CALCULATION_ERROR = "CALCULATION_ERROR"


@dataclass
class KcInfo:
    """Crop coefficient information with metadata."""
    value: float
    stage: str
    source: str
    verification_status: str
    evidence_level: str
    notes: str
    actual_study_location: str
    actual_soil_context: str
    actual_crop_system: str
    parameter_applicability: str


@dataclass
class ETcResult:
    """
    Result of crop evapotranspiration (ETc) calculation.
    
    ETc = ET₀ × Kc
    """
    # Main result
    etc_mm_day: Optional[float] = None
    
    # Input ET₀
    et0_mm_day: Optional[float] = None
    et0_method: Optional[str] = None
    et0_status: Optional[str] = None
    
    # Crop coefficient
    kc_value: Optional[float] = None
    kc_stage: Optional[str] = None
    kc_source: Optional[str] = None
    kc_verification: Optional[str] = None
    kc_evidence_level: Optional[str] = None
    kc_notes: Optional[str] = None
    kc_actual_study_location: Optional[str] = None
    kc_actual_soil_context: Optional[str] = None
    kc_actual_crop_system: Optional[str] = None
    kc_parameter_applicability: Optional[str] = None
    
    # Context
    crop: Optional[str] = None
    growth_stage: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    soil_type: Optional[str] = None
    season: Optional[str] = None
    
    # Status
    status: ETcStatus = ETcStatus.Kc_NOT_AVAILABLE
    missing_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if ETc was successfully calculated."""
        return (
            self.status == ETcStatus.CALCULATED 
            and self.etc_mm_day is not None
            and self.etc_mm_day >= 0
        )
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "etc_mm_day": self.etc_mm_day,
            "et0_mm_day": self.et0_mm_day,
            "et0_method": self.et0_method,
            "et0_status": self.et0_status,
            "kc_value": self.kc_value,
            "kc_stage": self.kc_stage,
            "kc_source": self.kc_source,
            "kc_verification": self.kc_verification,
            "kc_evidence_level": self.kc_evidence_level,
            "kc_notes": self.kc_notes,
            "kc_actual_study_location": self.kc_actual_study_location,
            "kc_actual_soil_context": self.kc_actual_soil_context,
            "kc_actual_crop_system": self.kc_actual_crop_system,
            "kc_parameter_applicability": self.kc_parameter_applicability,
            "crop": self.crop,
            "growth_stage": self.growth_stage,
            "state": self.state,
            "region": self.region,
            "soil_type": self.soil_type,
            "season": self.season,
            "status": self.status.value,
            "missing_inputs": self.missing_inputs,
            "warnings": self.warnings,
        }


VALID_GROWTH_STAGES = ["initial", "development", "mid", "late"]


def get_kc_for_crop_stage(
    crop: str,
    growth_stage: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    include_unverified: bool = False,
) -> Optional[KcInfo]:
    """
    Look up Kc value for a specific crop and growth stage from the database.
    
    Uses the existing threshold_config lookup functions.
    
    Args:
        crop: Crop name (e.g., "soybean")
        growth_stage: Growth stage ("initial", "development", "mid", "late")
        state: Indian state (optional, for context)
        region: Region within state (optional, for context)
        soil_type: Soil type (optional, for context)
        season: Growing season (optional, for context)
        include_unverified: If True, include unverified Kc values
    
    Returns:
        KcInfo object with Kc value and metadata, or None if not available
    """
    if growth_stage not in VALID_GROWTH_STAGES:
        return None
    
    config = get_threshold_config()
    kc_dict = config.get_crop_coefficients(
        crop=crop,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        include_unverified=include_unverified,
    )
    
    kc_key = f"kc_{growth_stage.lower()}"
    kc_value = kc_dict.get(kc_key)
    
    if kc_value is None:
        return None
    
    # Get the full record for metadata
    profile = config.get_crop_profile(
        crop=crop,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        growth_stage=growth_stage,
        include_unverified=include_unverified,
    )
    
    # Find the specific Kc record
    kc_rows = profile[
        (profile["parameter_type"] == "crop_coefficient")
        & (profile["growth_stage"].str.lower() == growth_stage.lower())
    ]
    
    if kc_rows.empty:
        # Fallback with minimal metadata
        return KcInfo(
            value=kc_value,
            stage=growth_stage,
            source="Database lookup (metadata unavailable)",
            verification_status="UNKNOWN",
            evidence_level="UNKNOWN",
            notes="",
            actual_study_location="",
            actual_soil_context="",
            actual_crop_system="",
            parameter_applicability="",
        )
    
    row = kc_rows.iloc[0]
    return KcInfo(
        value=kc_value,
        stage=growth_stage,
        source=str(row.get("source", "")),
        verification_status=str(row.get("verification_status", "")),
        evidence_level=str(row.get("evidence_level", "")),
        notes=str(row.get("notes", "")),
        actual_study_location=str(row.get("actual_study_location", "")),
        actual_soil_context=str(row.get("actual_soil_context", "")),
        actual_crop_system=str(row.get("actual_crop_system", "")),
        parameter_applicability=str(row.get("parameter_applicability", "")),
    )


def calculate_etc(
    et0_result: ET0Result,
    crop: str,
    growth_stage: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    include_unverified: bool = False,
) -> ETcResult:
    """
    Calculate crop evapotranspiration (ETc) = ET₀ × Kc.
    
    Looks up Kc from the threshold database for the given crop and growth stage.
    
    Args:
        et0_result: Valid ET0Result from Penman-Monteith or Hargreaves
        crop: Crop name (e.g., "soybean")
        growth_stage: Growth stage ("initial", "development", "mid", "late")
        state: Indian state (optional, for Kc context)
        region: Region within state (optional, for Kc context)
        soil_type: Soil type (optional, for Kc context)
        season: Growing season (optional, for Kc context)
        include_unverified: If True, include unverified Kc values
    
    Returns:
        ETcResult with ETc value and full metadata
    """
    result = ETcResult(
        crop=crop,
        growth_stage=growth_stage,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
    )
    warnings = []
    
    # Validate ET₀ result
    if et0_result is None:
        result.status = ETcStatus.INVALID_ET0
        result.missing_inputs = ["et0_result"]
        return result
    
    if not et0_result.is_valid():
        result.status = ETcStatus.INVALID_ET0
        result.et0_mm_day = et0_result.et0_mm_day
        result.et0_method = et0_result.method
        result.et0_status = et0_result.status.value
        result.missing_inputs = [f"ET₀ invalid: {et0_result.status.value}"]
        if et0_result.missing_inputs:
            result.missing_inputs.extend(et0_result.missing_inputs)
        return result
    
    et0 = et0_result.et0_mm_day
    if et0 is None:
        result.status = ETcStatus.INVALID_ET0
        result.missing_inputs = ["et0_mm_day is None"]
        return result
    
    # Validate ET₀ value
    if et0 < 0:
        result.status = ETcStatus.INVALID_ET0
        result.et0_mm_day = et0
        result.missing_inputs = ["ET₀ is negative"]
        return result
    
    if not (isinstance(et0, (int, float)) and et0 == et0):  # NaN check
        result.status = ETcStatus.INVALID_ET0
        result.et0_mm_day = et0
        result.missing_inputs = ["ET₀ is NaN"]
        return result
    
    # Store ET₀ info
    result.et0_mm_day = et0
    result.et0_method = et0_result.method
    result.et0_status = et0_result.status.value
    result.warnings.extend(et0_result.warnings)
    
    # Validate growth stage
    if growth_stage not in VALID_GROWTH_STAGES:
        result.status = ETcStatus.Kc_NOT_AVAILABLE
        result.missing_inputs = [f"Invalid growth_stage: {growth_stage}. Valid: {VALID_GROWTH_STAGES}"]
        return result
    
    # Look up Kc from database
    kc_info = get_kc_for_crop_stage(
        crop=crop,
        growth_stage=growth_stage,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        include_unverified=include_unverified,
    )
    
    if kc_info is None:
        result.status = ETcStatus.Kc_NOT_AVAILABLE
        result.missing_inputs = [f"No Kc available for {crop} at {growth_stage} stage with given context"]
        return result
    
    # Validate Kc
    if not (isinstance(kc_info.value, (int, float)) and kc_info.value == kc_info.value):  # NaN check
        result.status = ETcStatus.INVALID_KC
        result.missing_inputs = ["Kc is NaN"]
        return result
    
    if kc_info.value < 0:
        result.status = ETcStatus.INVALID_KC
        result.missing_inputs = ["Kc is negative"]
        return result
    
    if kc_info.value > 2.0:
        warnings.append(f"Kc unusually high: {kc_info.value:.2f} (typical range 0.1-1.5)")
    
    # Calculate ETc = ET₀ × Kc
    try:
        etc = et0 * kc_info.value
        
        if etc < 0:
            warnings.append(f"Calculated ETc negative ({etc:.3f}), set to 0")
            etc = 0.0
        
        if etc > 50:
            warnings.append(f"ETc unusually high: {etc:.2f} mm/day")
        
        # Populate result
        result.etc_mm_day = etc
        result.kc_value = kc_info.value
        result.kc_stage = kc_info.stage
        result.kc_source = kc_info.source
        result.kc_verification = kc_info.verification_status
        result.kc_evidence_level = kc_info.evidence_level
        result.kc_notes = kc_info.notes
        result.kc_actual_study_location = kc_info.actual_study_location
        result.kc_actual_soil_context = kc_info.actual_soil_context
        result.kc_actual_crop_system = kc_info.actual_crop_system
        result.kc_parameter_applicability = kc_info.parameter_applicability
        result.status = ETcStatus.CALCULATED
        result.warnings = warnings
        
    except Exception as e:
        result.status = ETcStatus.CALCULATION_ERROR
        result.missing_inputs = [f"calculation error: {str(e)}"]
    
    return result


def calculate_etc_from_weather(
    et0_result: ET0Result,
    crop: str,
    growth_stage: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
) -> ETcResult:
    """
    Convenience function to calculate ETc directly from ET₀ result.
    
    This is an alias for calculate_etc() with default parameters.
    """
    return calculate_etc(
        et0_result=et0_result,
        crop=crop,
        growth_stage=growth_stage,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        include_unverified=False,  # Default to verified only
    )