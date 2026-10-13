"""
Irrigation Assessment Pipeline - End-to-end integration of weather, ET₀, ETc, rainfall,
and soil-water decision modules.

This pipeline orchestrates the complete irrigation assessment workflow:
1. Weather input validation
2. ET₀ calculation (auto-selects Penman-Monteith or Hargreaves)
3. Kc lookup from database
4. ETc calculation (ET₀ × Kc)
5. Rainfall processing
6. Soil-water balance calculation via existing irrigation decision engine
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from .weather import WeatherInput, WeatherInputStatus
from .et0 import (
    ET0Result,
    ET0Status,
    calculate_et0_auto,
    penman_monteith_et0,
    hargreaves_et0,
)
from .etc import ETcResult, ETcStatus, calculate_etc
from .rainfall import RainfallResult, RainfallStatus, process_rainfall
from .irrigation_decision import (
    IrrigationDecision,
    IrrigationDecisionEngine,
    calculate_irrigation_need,
    EvidenceLevel,
    ThresholdType,
)


class PipelineStatus(Enum):
    """Overall pipeline execution status."""
    COMPLETE = "COMPLETE"
    PARTIALLY_AVAILABLE = "PARTIALLY_AVAILABLE"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED"
    ERROR = "ERROR"


@dataclass
class IrrigationAssessment:
    """
    Complete irrigation assessment result.
    
    Contains all inputs, intermediate calculations, and final irrigation decision.
    """
    # Inputs
    crop: str
    growth_stage: Optional[str] = None
    state: Optional[str] = None
    region: Optional[str] = None
    soil_type: Optional[str] = None
    season: Optional[str] = None
    current_soil_moisture: Optional[float] = None
    current_soil_moisture_unit: str = "fraction"
    user_mad: Optional[float] = None
    user_fc: Optional[float] = None
    user_pwp: Optional[float] = None
    user_root_zone_depth_cm: Optional[float] = None
    
    # Weather inputs
    air_temperature: Optional[float] = None
    minimum_temperature: Optional[float] = None
    maximum_temperature: Optional[float] = None
    relative_humidity: Optional[float] = None
    rainfall_mm: Optional[float] = None
    wind_speed: Optional[float] = None
    solar_radiation: Optional[float] = None
    latitude: Optional[float] = None
    elevation: Optional[float] = None
    timestamp: Optional[datetime] = None
    
    # Pipeline execution
    status: PipelineStatus = PipelineStatus.INSUFFICIENT_DATA
    missing_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Weather validation
    weather_status: Optional[WeatherInputStatus] = None
    weather_missing: List[str] = field(default_factory=list)
    weather_invalid: List[str] = field(default_factory=list)
    
    # ET₀
    et0_result: Optional[ET0Result] = None
    et0_mm_day: Optional[float] = None
    et0_method: Optional[str] = None
    et0_status: Optional[str] = None
    et0_missing_inputs: List[str] = field(default_factory=list)
    
    # ETc
    etc_result: Optional[ETcResult] = None
    etc_mm_day: Optional[float] = None
    kc_value: Optional[float] = None
    kc_stage: Optional[str] = None
    kc_source: Optional[str] = None
    kc_verification: Optional[str] = None
    kc_evidence_level: Optional[str] = None
    kc_missing: bool = False
    
    # Rainfall
    rainfall_result: Optional[RainfallResult] = None
    rainfall_mm: Optional[float] = None
    effective_rainfall_mm: Optional[float] = None
    effective_rainfall_status: Optional[str] = None
    
    # Soil-water / Irrigation decision
    irrigation_decision: Optional[IrrigationDecision] = None
    irrigation_required: Optional[bool] = None
    irrigation_status: Optional[str] = None
    threshold_used: Optional[float] = None
    threshold_type: Optional[str] = None
    threshold_parameter_type: Optional[str] = None
    calibration_required: bool = False
    calibration_notes: Optional[str] = None
    
    # Explanation
    explanation: str = ""
    
    def is_complete(self) -> bool:
        """Check if assessment produced a complete irrigation decision."""
        return (
            self.status in [PipelineStatus.COMPLETE, PipelineStatus.PARTIALLY_AVAILABLE]
            and self.irrigation_decision is not None
        )
    
    def has_et0(self) -> bool:
        """Check if ET₀ was calculated."""
        return self.et0_result is not None and self.et0_result.is_valid()
    
    def has_etc(self) -> bool:
        """Check if ETc was calculated."""
        return self.etc_result is not None and self.etc_result.is_valid()
    
    def has_rainfall(self) -> bool:
        """Check if rainfall was provided."""
        return self.rainfall_result is not None and self.rainfall_result.has_rainfall()
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization (legacy format)."""
        return {
            "inputs": {
                "crop": self.crop,
                "growth_stage": self.growth_stage,
                "state": self.state,
                "region": self.region,
                "soil_type": self.soil_type,
                "season": self.season,
                "current_soil_moisture": self.current_soil_moisture,
                "current_soil_moisture_unit": self.current_soil_moisture_unit,
                "user_mad": self.user_mad,
                "user_fc": self.user_fc,
                "user_pwp": self.user_pwp,
                "user_root_zone_depth_cm": self.user_root_zone_depth_cm,
            },
            "weather": {
                "air_temperature": self.air_temperature,
                "minimum_temperature": self.minimum_temperature,
                "maximum_temperature": self.maximum_temperature,
                "relative_humidity": self.relative_humidity,
                "rainfall_mm": self.rainfall_mm,
                "wind_speed": self.wind_speed,
                "solar_radiation": self.solar_radiation,
                "latitude": self.latitude,
                "elevation": self.elevation,
                "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            },
            "pipeline_status": self.status.value,
            "missing_inputs": self.missing_inputs,
            "warnings": self.warnings,
            "weather_validation": {
                "status": self.weather_status.value if self.weather_status else None,
                "missing": self.weather_missing,
                "invalid": self.weather_invalid,
            },
            "et0": {
                "et0_mm_day": self.et0_mm_day,
                "method": self.et0_method,
                "status": self.et0_status,
                "missing_inputs": self.et0_missing_inputs,
            },
            "etc": {
                "etc_mm_day": self.etc_mm_day,
                "kc_value": self.kc_value,
                "kc_stage": self.kc_stage,
                "kc_source": self.kc_source,
                "kc_verification": self.kc_verification,
                "kc_evidence_level": self.kc_evidence_level,
                "kc_missing": self.kc_missing,
            },
            "rainfall": {
                "rainfall_mm": self.rainfall_mm,
                "effective_rainfall_mm": self.effective_rainfall_mm,
                "effective_rainfall_status": self.effective_rainfall_status,
            },
            "irrigation_decision": self.irrigation_decision.to_dict() if self.irrigation_decision else None,
            "irrigation_required": self.irrigation_required,
            "irrigation_status": self.irrigation_status,
            "threshold_used": self.threshold_used,
            "threshold_type": self.threshold_type,
            "threshold_parameter_type": self.threshold_parameter_type,
            "calibration_required": self.calibration_required,
            "calibration_notes": self.calibration_notes,
            "explanation": self.explanation,
        }
    
    def to_final_dict(self) -> dict:
        """
        Convert to the final public API format.
        
        Returns structured result matching the required API:
        {
            "status": ...,
            "irrigation_required": ...,
            "crop": ...,
            "growth_stage": ...,
            "soil_water": {...},
            "weather": {...},
            "et0": {...},
            "crop_water_demand": {...},
            "rainfall": {...},
            "calibration": {...},
            "evidence": {...},
            "explanation": ...,
            "warnings": [...]
        }
        """
        # Soil water data from irrigation decision
        soil_water = {}
        if self.irrigation_decision:
            soil_water = {
                "current_soil_moisture": self.irrigation_decision.current_soil_moisture,
                "field_capacity": self.irrigation_decision.field_capacity,
                "permanent_wilting_point": self.irrigation_decision.permanent_wilting_point,
                "root_zone_depth_cm": self.irrigation_decision.root_zone_depth_cm,
                "depletion_mm": self.irrigation_decision.depletion_mm,
                "depletion_fraction": self.irrigation_decision.depletion_fraction,
            }
        
        # Weather data
        weather = {
            "temperature_c": self.air_temperature,
            "humidity_percent": self.relative_humidity,
            "rainfall_mm": self.rainfall_mm,
            "wind_speed_ms": self.wind_speed,
            "solar_radiation_mj_m2_day": self.solar_radiation,
        }
        
        # ET₀ data
        et0 = {
            "value_mm_day": self.et0_mm_day,
            "method": self.et0_method,
            "status": self.et0_status,
        }
        
        # Crop water demand
        crop_water_demand = {
            "kc": self.kc_value,
            "etc_mm_day": self.etc_mm_day,
            "growth_stage": self.growth_stage,
        }
        
        # Rainfall
        rainfall = {
            "rainfall_mm": self.rainfall_mm,
            "effective_rainfall_mm": self.effective_rainfall_mm,
            "effective_rainfall_status": self.effective_rainfall_status,
        }
        
        # Calibration
        calibration = {
            "required": self.calibration_required,
            "reason": self.calibration_notes if self.calibration_required else None,
        }
        
        # Evidence
        evidence = {}
        if self.irrigation_decision:
            evidence = {
                "source": self.irrigation_decision.threshold_source,
                "verification_status": self.irrigation_decision.threshold_verification_status,
                "evidence_level": self.irrigation_decision.threshold_evidence_level.value if self.irrigation_decision.threshold_evidence_level else None,
            }
        
        return {
            "status": self.status.value,
            "irrigation_required": self.irrigation_required,
            "crop": self.crop,
            "growth_stage": self.growth_stage,
            "soil_water": soil_water,
            "weather": weather,
            "et0": et0,
            "crop_water_demand": crop_water_demand,
            "rainfall": rainfall,
            "calibration": calibration,
            "evidence": evidence,
            "explanation": self.explanation,
            "warnings": self.warnings,
        }


def run_irrigation_assessment(
    crop: str,
    current_soil_moisture: float,
    current_soil_moisture_unit: str = "fraction",
    growth_stage: Optional[str] = None,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    user_mad: Optional[float] = None,
    user_fc: Optional[float] = None,
    user_pwp: Optional[float] = None,
    user_root_zone_depth_cm: Optional[float] = None,
    # Weather inputs
    air_temperature: Optional[float] = None,
    minimum_temperature: Optional[float] = None,
    maximum_temperature: Optional[float] = None,
    relative_humidity: Optional[float] = None,
    rainfall_mm: Optional[float] = None,
    wind_speed: Optional[float] = None,
    solar_radiation: Optional[float] = None,
    latitude: Optional[float] = None,
    elevation: Optional[float] = None,
    timestamp: Optional[datetime] = None,
) -> IrrigationAssessment:
    """
    Run complete irrigation assessment pipeline.
    
    Args:
        crop: Crop name (e.g., "soybean")
        current_soil_moisture: Current soil moisture (VWC fraction or percent)
        current_soil_moisture_unit: "fraction" or "percent"
        growth_stage: Crop growth stage ("initial", "development", "mid", "late")
        state: Indian state
        region: Region within state
        soil_type: Soil type
        season: Growing season
        user_mad: User-supplied MAD override
        user_fc: User-supplied FC override
        user_pwp: User-supplied PWP override
        user_root_zone_depth_cm: User-supplied root zone depth
        air_temperature: Mean air temperature (°C)
        minimum_temperature: Minimum air temperature (°C)
        maximum_temperature: Maximum air temperature (°C)
        relative_humidity: Mean relative humidity (%)
        rainfall_mm: Rainfall (mm)
        wind_speed: Wind speed at 2m (m/s)
        solar_radiation: Solar radiation (MJ/m²/day)
        latitude: Latitude (degrees, positive north)
        elevation: Elevation (m)
        timestamp: Date/time for day-of-year calculation
    
    Returns:
        IrrigationAssessment with complete results
    """
    # Initialize assessment
    assessment = IrrigationAssessment(
        crop=crop,
        growth_stage=growth_stage,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        current_soil_moisture=current_soil_moisture,
        current_soil_moisture_unit=current_soil_moisture_unit,
        user_mad=user_mad,
        user_fc=user_fc,
        user_pwp=user_pwp,
        user_root_zone_depth_cm=user_root_zone_depth_cm,
        air_temperature=air_temperature,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        relative_humidity=relative_humidity,
        rainfall_mm=rainfall_mm,
        wind_speed=wind_speed,
        solar_radiation=solar_radiation,
        latitude=latitude,
        elevation=elevation,
        timestamp=timestamp,
    )
    
    # Step 1: Validate weather inputs
    weather_input = WeatherInput(
        timestamp=timestamp or datetime.now(),
        air_temperature=air_temperature,
        minimum_temperature=minimum_temperature,
        maximum_temperature=maximum_temperature,
        relative_humidity=relative_humidity,
        rainfall_mm=rainfall_mm,
        wind_speed=wind_speed,
        solar_radiation=solar_radiation,
        latitude=latitude,
        elevation=elevation,
    )
    
    assessment.weather_status = weather_input.status
    assessment.weather_missing = weather_input.missing_inputs
    assessment.weather_invalid = weather_input.invalid_inputs
    assessment.warnings.extend(weather_input.warnings)
    
    if weather_input.invalid_inputs:
        assessment.status = PipelineStatus.INVALID_INPUT
        assessment.missing_inputs.extend(weather_input.invalid_inputs)
        return assessment
    
    # Step 2: Calculate ET₀ (auto-select method)
    et0_result = calculate_et0_auto(weather_input)
    assessment.et0_result = et0_result
    assessment.et0_mm_day = et0_result.et0_mm_day
    assessment.et0_method = et0_result.method
    assessment.et0_status = et0_result.status.value
    assessment.et0_missing_inputs = et0_result.missing_inputs
    assessment.warnings.extend(et0_result.warnings)
    
    if not et0_result.is_valid():
        if et0_result.status == ET0Status.INSUFFICIENT_DATA:
            assessment.status = PipelineStatus.INSUFFICIENT_DATA
            assessment.missing_inputs.extend([f"ET₀: {m}" for m in et0_result.missing_inputs])
        else:
            assessment.status = PipelineStatus.INVALID_INPUT
            assessment.missing_inputs.extend([f"ET₀: {m}" for m in et0_result.missing_inputs])
        # Continue to soil-water decision even without ET₀
    else:
        # Step 3: Calculate ETc if ET₀ is available
        if growth_stage:
            etc_result = calculate_etc(
                et0_result=et0_result,
                crop=crop,
                growth_stage=growth_stage,
                state=state,
                region=region,
                soil_type=soil_type,
                season=season,
                include_unverified=False,
            )
            assessment.etc_result = etc_result
            assessment.etc_mm_day = etc_result.etc_mm_day
            assessment.kc_value = etc_result.kc_value
            assessment.kc_stage = etc_result.kc_stage
            assessment.kc_source = etc_result.kc_source
            assessment.kc_verification = etc_result.kc_verification
            assessment.kc_evidence_level = etc_result.kc_evidence_level
            assessment.warnings.extend(etc_result.warnings)
            
            if etc_result.status == ETcStatus.Kc_NOT_AVAILABLE:
                assessment.kc_missing = True
                assessment.warnings.append(f"Kc not available for {crop} at {growth_stage} stage")
            elif etc_result.status == ETcStatus.INVALID_KC:
                assessment.warnings.append(f"Invalid Kc value for {crop}")
        else:
            assessment.warnings.append("No growth_stage provided - ETc not calculated")
    
    # Step 4: Process rainfall
    rainfall_result = process_rainfall(rainfall_mm)
    assessment.rainfall_result = rainfall_result
    assessment.rainfall_mm = rainfall_result.rainfall_mm
    assessment.effective_rainfall_mm = rainfall_result.effective_rainfall_mm
    assessment.effective_rainfall_status = rainfall_result.effective_rainfall_status.value
    assessment.warnings.extend(rainfall_result.warnings)
    
    if rainfall_result.status == RainfallStatus.MISSING:
        assessment.warnings.append("Rainfall data not available")
    elif rainfall_result.status == RainfallStatus.INVALID:
        assessment.warnings.append(f"Invalid rainfall: {rainfall_result.invalid_reason}")
    
    # Step 5: Run existing irrigation decision engine
    try:
        irrigation_decision = calculate_irrigation_need(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            growth_stage=growth_stage,
            current_soil_moisture=current_soil_moisture,
            current_soil_moisture_unit=current_soil_moisture_unit,
            user_mad=user_mad,
            user_fc=user_fc,
            user_pwp=user_pwp,
            user_root_zone_depth_cm=user_root_zone_depth_cm,
        )
        
        assessment.irrigation_decision = irrigation_decision
        assessment.irrigation_required = irrigation_decision.irrigation_required
        assessment.irrigation_status = irrigation_decision.status
        assessment.threshold_used = irrigation_decision.threshold_used
        assessment.threshold_type = irrigation_decision.threshold_type.value if irrigation_decision.threshold_type else None
        assessment.threshold_parameter_type = irrigation_decision.threshold_parameter_type
        assessment.calibration_required = irrigation_decision.calibration_required
        assessment.calibration_notes = irrigation_decision.calibration_notes
        assessment.explanation = irrigation_decision.explanation
        
        # Check if calibration is required (e.g., MP soybean Vertisol MAD)
        if irrigation_decision.calibration_required:
            assessment.status = PipelineStatus.CALIBRATION_REQUIRED
        elif et0_result.is_valid() and (not growth_stage or assessment.etc_result and assessment.etc_result.is_valid()):
            assessment.status = PipelineStatus.COMPLETE
        else:
            assessment.status = PipelineStatus.PARTIALLY_AVAILABLE
            
    except ValueError as e:
        assessment.status = PipelineStatus.INVALID_INPUT
        assessment.missing_inputs.append(str(e))
        assessment.warnings.append(f"Irrigation decision error: {str(e)}")
    
    # Add integration-specific warnings
    if assessment.effective_rainfall_status == "NOT_CALCULATED":
        assessment.warnings.append("Effective rainfall NOT_CALCULATED - measured rainfall not automatically converted to effective rainfall")
    
    if assessment.kc_missing:
        assessment.warnings.append("Crop coefficient (Kc) unavailable - ETc not calculated")
    
    if not et0_result.is_valid() and et0_result.status == ET0Status.INSUFFICIENT_DATA:
        assessment.warnings.append(f"ET₀ unavailable - insufficient weather data: {', '.join(et0_result.missing_inputs)}")
    
    return assessment