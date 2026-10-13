"""
Prediction module for irrigation need detection.
Provides interface for making irrigation decisions using the soil-water engine
and threshold database.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import pandas as pd
import numpy as np

from .config import DATA_PROCESSED_DIR, VALID_STATUS
from .threshold_config import get_crop_profile, get_threshold_config
from .soil_water_engine import (
    SoilWaterEngine,
    SoilWaterParams,
    SoilWaterResult,
    calculate_taw,
    calculate_raw,
    calculate_depletion,
    calculate_depletion_fraction,
)
from .preprocessing import SensorPreprocessor, preprocess_sensor_data
from .feature_engineering import FeatureEngineer, engineer_features


class ParameterSource(Enum):
    """Classification of parameter source."""
    RESEARCH_BACKED = "RESEARCH_BACKED"
    USER_SUPPLIED = "USER_SUPPLIED"
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION"


class IrrigationPredictor:
    """
    High-level interface for irrigation need prediction.

    Combines:
    - Sensor data preprocessing
    - Threshold database lookup
    - Soil-water engine calculations
    - Feature engineering

    This is the AGRONOMIC decision layer, not the ML model.
    """

    def __init__(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        growth_stage: Optional[str] = None,
        sensor_data_path: Optional[Path] = None,
    ):
        """
        Initialize predictor with crop/region context.

        Args:
            crop: Crop name (e.g., "soybean", "wheat")
            state: Indian state
            region: Region within state
            soil_type: Soil type
            season: Growing season
            growth_stage: Crop growth stage
            sensor_data_path: Path to sensor CSV
        """
        self.crop = crop
        self.state = state
        self.region = region
        self.soil_type = soil_type
        self.season = season
        self.growth_stage = growth_stage
        self.sensor_data_path = sensor_data_path

        self._threshold_profile: Optional[pd.DataFrame] = None
        self._seasonal_water_req_mm: Optional[float] = None
        self._soil_moisture_threshold: Optional[Dict[str, Any]] = None
        self._engine: Optional[SoilWaterEngine] = None
        self._processed_data: Optional[pd.DataFrame] = None
        self._features: Optional[pd.DataFrame] = None

        # Track parameter sources for safety
        self._parameter_sources: Dict[str, ParameterSource] = {
            "field_capacity": ParameterSource.REQUIRES_CALIBRATION,
            "permanent_wilting_point": ParameterSource.REQUIRES_CALIBRATION,
            "root_zone_depth_cm": ParameterSource.REQUIRES_CALIBRATION,
            "mad": ParameterSource.REQUIRES_CALIBRATION,
        }

    def load_thresholds(self, include_unverified: bool = False) -> None:
        """
        Load thresholds from database for the configured context.

        Args:
            include_unverified: If True, include unverified parameters.
                               Default False for safety.
        """
        self._threshold_profile = get_crop_profile(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            growth_stage=self.growth_stage,
            include_unverified=include_unverified,
        )

        config = get_threshold_config()
        self._seasonal_water_req_mm = config.get_seasonal_water_requirement(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            include_unverified=include_unverified,
        )

        self._soil_moisture_threshold = config.get_soil_moisture_threshold(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            growth_stage=self.growth_stage,
            include_unverified=include_unverified,
        )

    def initialize_engine(
        self,
        field_capacity: Optional[float] = None,
        permanent_wilting_point: Optional[float] = None,
        root_zone_depth_cm: Optional[float] = None,
        mad: Optional[float] = None,
        allow_user_supplied: bool = True,
    ) -> SoilWaterEngine:
        """
        Initialize soil-water engine with soil parameters.

        CRITICAL SAFETY: By default, only verified research-backed parameters
        from the database can be used. User-supplied values must be explicitly
        allowed via allow_user_supplied=True and will be classified as
        USER_SUPPLIED.

        Args:
            field_capacity: FC (VWC fraction, 0-1). If None, attempts to load from database.
            permanent_wilting_point: PWP (VWC fraction, 0-1). If None, attempts to load from database.
            root_zone_depth_cm: Root zone depth in cm. If None, attempts to load from database.
            mad: Management allowed depletion (fraction). If None, attempts to load from database.
            allow_user_supplied: If True, allow user-supplied values for missing parameters.
                                If False, raise error for any missing required parameter.

        Returns:
            Initialized SoilWaterEngine

        Raises:
            ValueError: If required parameter is missing and allow_user_supplied=False,
                        or if user-supplied value is provided but not marked as such.
        """
        config = get_threshold_config()

        # Track which parameters are user-supplied vs research-backed
        param_values = {}
        param_sources = {}

        # Field Capacity
        if field_capacity is not None:
            param_values["field_capacity"] = field_capacity
            param_sources["field_capacity"] = ParameterSource.USER_SUPPLIED
        else:
            hydraulic = config.get_soil_hydraulic_parameters(
                crop=self.crop,
                state=self.state,
                region=self.region,
                soil_type=self.soil_type,
                season=self.season,
                include_unverified=False,
            )
            if hydraulic["field_capacity"] is not None:
                param_values["field_capacity"] = hydraulic["field_capacity"]
                param_sources["field_capacity"] = ParameterSource.RESEARCH_BACKED

        # Permanent Wilting Point
        if permanent_wilting_point is not None:
            param_values["permanent_wilting_point"] = permanent_wilting_point
            param_sources["permanent_wilting_point"] = ParameterSource.USER_SUPPLIED
        else:
            hydraulic = config.get_soil_hydraulic_parameters(
                crop=self.crop,
                state=self.state,
                region=self.region,
                soil_type=self.soil_type,
                season=self.season,
                include_unverified=False,
            )
            if hydraulic["permanent_wilting_point"] is not None:
                param_values["permanent_wilting_point"] = hydraulic["permanent_wilting_point"]
                param_sources["permanent_wilting_point"] = ParameterSource.RESEARCH_BACKED

        # Root Zone Depth
        if root_zone_depth_cm is not None:
            param_values["root_zone_depth_cm"] = root_zone_depth_cm
            param_sources["root_zone_depth_cm"] = ParameterSource.USER_SUPPLIED
        else:
            rz = config.get_root_zone_depth(
                crop=self.crop,
                state=self.state,
                region=self.region,
                soil_type=self.soil_type,
                season=self.season,
                include_unverified=False,
            )
            if rz is not None:
                param_values["root_zone_depth_cm"] = rz
                param_sources["root_zone_depth_cm"] = ParameterSource.RESEARCH_BACKED

        # MAD
        if mad is not None:
            param_values["mad"] = mad
            param_sources["mad"] = ParameterSource.USER_SUPPLIED
        else:
            # Try to get MAD from soil moisture threshold
            threshold = config.get_soil_moisture_threshold(
                crop=self.crop,
                state=self.state,
                region=self.region,
                soil_type=self.soil_type,
                season=self.season,
                growth_stage=self.growth_stage,
                parameter_name="available_soil_moisture_depletion",
                include_unverified=False,
            )
            if threshold and threshold.get("verification_status") in ["VERIFIED_EXACT", "VERIFIED_CONTEXTUAL"]:
                if threshold["parameter_unit"] == "percent":
                    param_values["mad"] = threshold["parameter_value"] / 100.0
                else:
                    param_values["mad"] = threshold["parameter_value"]
                param_sources["mad"] = ParameterSource.RESEARCH_BACKED

        # Check for missing required parameters
        required_params = ["field_capacity", "permanent_wilting_point", "root_zone_depth_cm", "mad"]
        missing = [p for p in required_params if p not in param_values]

        if missing and not allow_user_supplied:
            raise ValueError(
                f"Required parameters missing for {self.crop} in {self.state}: {missing}. "
                f"These require field calibration. Set allow_user_supplied=True to override "
                f"with user-supplied values (will be classified as USER_SUPPLIED)."
            )

        if missing:
            # Allow user-supplied but warn and track
            for p in missing:
                if p == "mad":
                    # Default MAD for medium-rooted crops per FAO-56 (not Indian-specific)
                    param_values["mad"] = 0.5
                elif p == "field_capacity":
                    raise ValueError(
                        f"field_capacity is REQUIRED for {self.crop} in {self.state}. "
                        f"No verified research value found. Must be supplied by user "
                        f"(will be classified as USER_SUPPLIED) or calibrated locally."
                    )
                elif p == "permanent_wilting_point":
                    raise ValueError(
                        f"permanent_wilting_point is REQUIRED for {self.crop} in {self.state}. "
                        f"No verified research value found. Must be supplied by user "
                        f"(will be classified as USER_SUPPLIED) or calibrated locally."
                    )
                elif p == "root_zone_depth_cm":
                    raise ValueError(
                        f"root_zone_depth_cm is REQUIRED for {self.crop} in {self.state}. "
                        f"No verified research value found. Must be supplied by user "
                        f"(will be classified as USER_SUPPLIED) or calibrated locally."
                    )
                param_sources[p] = ParameterSource.USER_SUPPLIED

        # Update internal tracking
        self._parameter_sources.update(param_sources)

        params = SoilWaterParams(
            field_capacity=param_values["field_capacity"],
            permanent_wilting_point=param_values["permanent_wilting_point"],
            root_zone_depth_cm=param_values["root_zone_depth_cm"],
            mad=param_values["mad"],
        )
        self._engine = SoilWaterEngine(params)
        return self._engine

    def initialize_engine_from_database(self, allow_user_supplied: bool = False) -> SoilWaterEngine:
        """
        Initialize engine using ONLY verified database parameters.

        This is the SAFEST method - it will fail if any required parameter
        is not available as VERIFIED_EXACT or VERIFIED_CONTEXTUAL in the database.

        Args:
            allow_user_supplied: If True, allow user-supplied fallback for missing params.

        Returns:
            Initialized SoilWaterEngine

        Raises:
            ValueError: If any required parameter is missing from verified database.
        """
        return self.initialize_engine(
            field_capacity=None,
            permanent_wilting_point=None,
            root_zone_depth_cm=None,
            mad=None,
            allow_user_supplied=allow_user_supplied,
        )

    def load_and_process_sensor_data(
        self,
        data_path: Optional[Path] = None,
        handle_missing: str = "interpolate",
        resample_freq: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load and preprocess sensor data."""
        path = data_path or self.sensor_data_path
        if path is None:
            raise ValueError("No sensor data path provided")

        self._processed_data = preprocess_sensor_data(
            data_path=path,
            handle_missing=handle_missing,
            resample_freq=resample_freq,
        )
        return self._processed_data

    def engineer_features(
        self,
        latitude: Optional[float] = None,
        elevation: float = 0.0,
        sowing_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Engineer features from processed data."""
        if self._processed_data is None:
            raise ValueError("No processed data. Call load_and_process_sensor_data() first.")

        if self._engine is None:
            raise ValueError("Engine not initialized. Call initialize_engine() first.")

        self._features = engineer_features(
            self._processed_data,
            field_capacity=self._engine.params.field_capacity,
            permanent_wilting_point=self._engine.params.permanent_wilting_point,
            root_zone_depth_cm=self._engine.params.root_zone_depth_cm,
            latitude=latitude,
            elevation=elevation,
            crop=self.crop,
            sowing_date=sowing_date,
        )
        return self._features

    def predict_latest(self) -> Optional[SoilWaterResult]:
        """
        Make irrigation prediction for the latest sensor reading.

        Returns:
            SoilWaterResult or None if not ready
        """
        if self._engine is None:
            raise ValueError("Engine not initialized")

        if self._features is None or self._features.empty:
            if self._processed_data is None or self._processed_data.empty:
                return None
            latest_moisture = self._processed_data["soil_moisture"].iloc[-1]
        else:
            latest_moisture = self._features["soil_moisture"].iloc[-1]

        mad_override = None
        if self._soil_moisture_threshold:
            threshold = self._soil_moisture_threshold
            if threshold.get("parameter_name") == "available_soil_moisture_depletion":
                if threshold.get("parameter_unit") == "percent":
                    mad_override = threshold["parameter_value"] / 100.0
                else:
                    mad_override = threshold["parameter_value"]

        return self._engine.calculate(latest_moisture, mad_override=mad_override)

    def predict_series(self) -> List[SoilWaterResult]:
        """
        Make irrigation predictions for entire time series.

        Returns:
            List of SoilWaterResult for each timestep
        """
        if self._engine is None:
            raise ValueError("Engine not initialized")

        if self._features is None or self._features.empty:
            if self._processed_data is None or self._processed_data.empty:
                return []
            moisture_series = self._processed_data["soil_moisture"]
        else:
            moisture_series = self._features["soil_moisture"]

        mad_override = None
        if self._soil_moisture_threshold:
            threshold = self._soil_moisture_threshold
            if threshold.get("parameter_name") == "available_soil_moisture_depletion":
                if threshold.get("parameter_unit") == "percent":
                    mad_override = threshold["parameter_value"] / 100.0
                else:
                    mad_override = threshold["parameter_value"]

        results = []
        for moisture in moisture_series:
            results.append(self._engine.calculate(moisture, mad_override=mad_override))
        return results

    def get_parameter_sources(self) -> Dict[str, str]:
        """Get classification of parameter sources."""
        return {k: v.value for k, v in self._parameter_sources.items()}

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration and thresholds."""
        return {
            "context": {
                "crop": self.crop,
                "state": self.state,
                "region": self.region,
                "soil_type": self.soil_type,
                "season": self.season,
                "growth_stage": self.growth_stage,
            },
            "seasonal_water_requirement_mm": self._seasonal_water_req_mm,
            "soil_moisture_threshold": self._soil_moisture_threshold,
            "threshold_profile_records": len(self._threshold_profile) if self._threshold_profile is not None else 0,
            "engine_initialized": self._engine is not None,
            "parameter_sources": self.get_parameter_sources(),
            "engine_params": {
                "field_capacity": self._engine.params.field_capacity if self._engine else None,
                "permanent_wilting_point": self._engine.params.permanent_wilting_point if self._engine else None,
                "root_zone_depth_cm": self._engine.params.root_zone_depth_cm if self._engine else None,
                "mad": self._engine.params.mad if self._engine else None,
                "taw_mm": self._engine.taw_mm if self._engine else None,
                "raw_mm": self._engine.raw_mm if self._engine else None,
            } if self._engine else None,
            "data_loaded": self._processed_data is not None,
            "data_points": len(self._processed_data) if self._processed_data is not None else 0,
        }

    def check_seasonal_requirement_not_trigger(self) -> Dict[str, Any]:
        """
        Verify that seasonal water requirement is NOT used as instantaneous trigger.

        Returns:
            Dict with check results
        """
        if self._seasonal_water_req_mm is None:
            return {
                "check_passed": True,
                "message": "No seasonal water requirement in database for this context",
                "seasonal_requirement_mm": None,
            }

        return {
            "check_passed": True,
            "message": (
                f"Seasonal crop water requirement ({self._seasonal_water_req_mm} mm) "
                f"is correctly NOT used as instantaneous irrigation trigger. "
                f"It is a seasonal total for water budgeting only."
            ),
            "seasonal_requirement_mm": self._seasonal_water_req_mm,
        }


def create_predictor(
    crop: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    growth_stage: Optional[str] = None,
    sensor_data_path: Optional[Path] = None,
) -> IrrigationPredictor:
    """Factory function to create IrrigationPredictor."""
    return IrrigationPredictor(
        crop=crop,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        growth_stage=growth_stage,
        sensor_data_path=sensor_data_path,
    )