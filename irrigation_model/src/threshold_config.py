"""
Crop/Region configuration for irrigation thresholds.
Provides lookup functions for research-backed parameters.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd

from .threshold_database import ThresholdDatabase, get_database


class ThresholdConfig:
    """Configuration loader for crop/region specific irrigation thresholds."""

    def __init__(self, database_path: Optional[Path] = None):
        self.db = get_database()

    def get_crop_profile(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        growth_stage: Optional[str] = None,
        include_unverified: bool = False,
        verification_status: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Get irrigation thresholds matching the specified criteria.

        Returns only parameters supported by the database.
        Does not invent FC/PWP/root depth values.

        Args:
            crop: Crop name (e.g., "soybean", "wheat")
            state: Indian state (e.g., "Madhya Pradesh", "Bihar")
            region: Region within state (e.g., "Bhopal/Indore region")
            soil_type: Soil type (e.g., "Vertisol")
            season: Growing season (e.g., "Kharif", "Rabi")
            growth_stage: Crop growth stage
            include_unverified: If False (default), only return VERIFIED_EXACT and
                                VERIFIED_CONTEXTUAL parameters.
                                If True, also return NOT_VERIFIED, DERIVED,
                                REQUIRES_FIELD_CALIBRATION.
            verification_status: Optional list of specific verification statuses
                                 to include. Overrides include_unverified.

        Returns:
            DataFrame with matching threshold records
        """
        if verification_status is None:
            if include_unverified:
                verification_status = [
                    "VERIFIED_EXACT",
                    "VERIFIED_CONTEXTUAL",
                    "DERIVED",
                    "NOT_VERIFIED",
                    "REQUIRES_FIELD_CALIBRATION",
                ]
            else:
                verification_status = ["VERIFIED_EXACT", "VERIFIED_CONTEXTUAL"]

        results = []
        for status in verification_status:
            df = self.db.query(
                crop=crop,
                state=state,
                region=region,
                soil_type=soil_type,
                season=season,
                verification_status=status,
                growth_stage=growth_stage,
            )
            if not df.empty:
                results.append(df)

        if results:
            return pd.concat(results, ignore_index=True)
        return pd.DataFrame()

    def get_seasonal_water_requirement(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        include_unverified: bool = False,
    ) -> Optional[float]:
        """
        Get seasonal crop water requirement (mm) if available.

        IMPORTANT: This is a seasonal total, NOT an instantaneous irrigation trigger.

        Args:
            include_unverified: If False (default), only return verified values.

        Returns:
            Water requirement in mm, or None if not available
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            include_unverified=include_unverified,
        )

        if profile.empty:
            return None

        swr = profile[
            profile["parameter_type"] == "seasonal_crop_water_requirement"
        ]["parameter_value"]

        if swr.empty:
            return None

        return float(swr.iloc[0])

    def get_soil_moisture_threshold(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        growth_stage: Optional[str] = None,
        parameter_name: str = "available_soil_moisture_depletion",
        include_unverified: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get soil moisture depletion threshold for irrigation trigger.

        Args:
            parameter_name: Specific parameter name to retrieve
            include_unverified: If False (default), only return verified values.

        Returns:
            Dict with parameter_value, parameter_unit, evidence_level, verification_status, notes, source
            or None if not available
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            growth_stage=growth_stage,
            include_unverified=include_unverified,
        )

        if profile.empty:
            return None

        matches = profile[profile["parameter_name"] == parameter_name]

        if matches.empty:
            return None

        row = matches.iloc[0]
        return {
            "parameter_value": row["parameter_value"],
            "parameter_unit": row["parameter_unit"],
            "evidence_level": row["evidence_level"],
            "verification_status": row["verification_status"],
            "notes": row["notes"],
            "source": row["source"],
            "doi": row["doi"],
        }

    def get_soil_hydraulic_parameters(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        include_unverified: bool = False,
    ) -> Dict[str, Optional[float]]:
        """
        Get soil hydraulic parameters (FC, PWP) for the specified context.

        Returns:
            Dict with field_capacity, permanent_wilting_point (as fractions 0-1)
            Values are None if not available/verified.
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            include_unverified=include_unverified,
        )

        result = {
            "field_capacity": None,
            "permanent_wilting_point": None,
        }

        if profile.empty:
            return result

        fc_rows = profile[
            (profile["parameter_type"] == "soil_hydraulic_parameter")
            & (profile["parameter_name"] == "field_capacity")
        ]
        if not fc_rows.empty:
            result["field_capacity"] = float(fc_rows.iloc[0]["parameter_value"]) / 100.0

        pwp_rows = profile[
            (profile["parameter_type"] == "soil_hydraulic_parameter")
            & (profile["parameter_name"] == "permanent_wilting_point")
        ]
        if not pwp_rows.empty:
            result["permanent_wilting_point"] = float(pwp_rows.iloc[0]["parameter_value"]) / 100.0

        return result

    def get_root_zone_depth(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        include_unverified: bool = False,
    ) -> Optional[float]:
        """
        Get root zone depth (cm) for the specified context.

        Returns:
            Root zone depth in cm, or None if not available/verified.
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            include_unverified=include_unverified,
        )

        if profile.empty:
            return None

        rz_rows = profile[
            (profile["parameter_type"] == "root_zone_parameter")
            & (profile["parameter_name"] == "root_zone_depth")
        ]
        if rz_rows.empty:
            return None

        return float(rz_rows.iloc[0]["parameter_value"])

    def get_crop_coefficients(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        include_unverified: bool = False,
    ) -> Dict[str, Optional[float]]:
        """
        Get crop coefficients (Kc) by growth stage.

        Returns:
            Dict with kc_initial, kc_development, kc_mid, kc_late
            Values are None if not available/verified.
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            include_unverified=include_unverified,
        )

        result = {
            "kc_initial": None,
            "kc_development": None,
            "kc_mid": None,
            "kc_late": None,
        }

        if profile.empty:
            return result

        kc_rows = profile[profile["parameter_type"] == "crop_coefficient"]
        if kc_rows.empty:
            return result

        for _, row in kc_rows.iterrows():
            stage = row["growth_stage"]
            if stage == "initial":
                result["kc_initial"] = float(row["parameter_value"])
            elif stage == "development":
                result["kc_development"] = float(row["parameter_value"])
            elif stage == "mid":
                result["kc_mid"] = float(row["parameter_value"])
            elif stage == "late":
                result["kc_late"] = float(row["parameter_value"])

        return result

    def get_et0_methodology(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        include_unverified: bool = False,
    ) -> Dict[str, Any]:
        """
        Get ET0 methodology information.

        Returns:
            Dict with available methods and notes
        """
        profile = self.get_crop_profile(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            include_unverified=include_unverified,
        )

        result = {
            "methods": [],
            "notes": "",
        }

        if profile.empty:
            return result

        et0_rows = profile[profile["parameter_type"] == "et0_methodology"]
        for _, row in et0_rows.iterrows():
            if row["parameter_value"] == 1:
                result["methods"].append(row["parameter_name"])
        result["notes"] = et0_rows.iloc[0]["notes"] if not et0_rows.empty else ""

        return result

    def get_parameters_requiring_calibration(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get all parameters that require field calibration for the context."""
        return self.db.get_parameters_requiring_calibration(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
        )

    def list_available_crops(self) -> List[str]:
        """List all crops in the database."""
        return self.db.list_available_crops()

    def list_available_states(self, crop: Optional[str] = None) -> List[str]:
        """List all states, optionally filtered by crop."""
        return self.db.list_available_states(crop)

    def list_available_soil_types(self, crop: Optional[str] = None) -> List[str]:
        """List all soil types, optionally filtered by crop."""
        return self.db.list_available_soil_types(crop)


_threshold_config_instance: Optional[ThresholdConfig] = None


def get_threshold_config() -> ThresholdConfig:
    """Get singleton instance of ThresholdConfig."""
    global _threshold_config_instance
    if _threshold_config_instance is None:
        _threshold_config_instance = ThresholdConfig()
    return _threshold_config_instance


def get_crop_profile(
    crop: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    growth_stage: Optional[str] = None,
    include_unverified: bool = False,
) -> pd.DataFrame:
    """Convenience function to get crop profile."""
    return get_threshold_config().get_crop_profile(
        crop=crop,
        state=state,
        region=region,
        soil_type=soil_type,
        season=season,
        growth_stage=growth_stage,
        include_unverified=include_unverified,
    )