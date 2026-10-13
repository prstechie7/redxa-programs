"""
Feature engineering for irrigation prediction.
Creates agronomically meaningful features from sensor data.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

from .config import UNITS


class FeatureEngineer:
    """Creates irrigation-relevant features from preprocessed sensor data."""

    def __init__(self):
        pass

    def compute_vpd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute Vapor Pressure Deficit (VPD) from air temperature and humidity.

        VPD = es - ea
        es = 0.6108 * exp(17.27 * T / (T + 237.3))  [kPa]
        ea = es * RH / 100  [kPa]

        Args:
            df: DataFrame with air_temperature (C) and air_humidity (%)

        Returns:
            DataFrame with added vpd_kpa column

        Raises:
            ValueError: If air_temperature or air_humidity columns are missing
        """
        df = df.copy()

        if "air_temperature" not in df.columns:
            raise ValueError("air_temperature column is required for VPD calculation")
        if "air_humidity" not in df.columns:
            raise ValueError("air_humidity column is required for VPD calculation")

        T = df["air_temperature"]
        RH = df["air_humidity"]

        es = 0.6108 * np.exp((17.27 * T) / (T + 237.3))
        ea = es * RH / 100.0
        df["vpd_kpa"] = es - ea

        return df

    def compute_et0_hargreaves(
        self,
        df: pd.DataFrame,
        latitude: float,
        elevation: float = 0.0,
    ) -> pd.DataFrame:
        """
        Compute reference evapotranspiration (ET0) using Hargreaves equation.

        ET0 = 0.0023 * Ra * (Tmax - Tmin)^0.5 * (Tmean + 17.8)

        Args:
            df: DataFrame with air_temperature (daily max/min/mean)
            latitude: Site latitude in degrees
            elevation: Site elevation in meters

        Returns:
            DataFrame with added et0_mm column

        Raises:
            ValueError: If required inputs (latitude, air_temperature) are missing
        """
        if latitude is None:
            raise ValueError("latitude is required for ET0 calculation")

        df = df.copy()

        required = ["tmax", "tmin", "tmean"]
        if not all(c in df.columns for c in required):
            if "air_temperature" in df.columns and "timestamp" in df.columns:
                daily = df.set_index("timestamp").resample("D")["air_temperature"]
                df["tmax"] = daily.transform("max")
                df["tmin"] = daily.transform("min")
                df["tmean"] = daily.transform("mean")
            else:
                raise ValueError(
                    "ET0 calculation requires either (tmax, tmin, tmean) columns "
                    "or (air_temperature + timestamp) columns"
                )

        lat_rad = np.deg2rad(latitude)

        doy = df["timestamp"].dt.dayofyear if "timestamp" in df.columns else np.arange(len(df)) + 1

        dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
        delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
        ws = np.arccos(-np.tan(lat_rad) * np.tan(delta))
        ra = (24 * 60 / np.pi) * 0.0820 * dr * (
            ws * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )

        tmax = df["tmax"]
        tmin = df["tmin"]
        tmean = df["tmean"]

        df["et0_mm"] = 0.0023 * ra * np.sqrt(tmax - tmin) * (tmean + 17.8)
        df["et0_mm"] = df["et0_mm"].clip(lower=0)

        return df

    def compute_soil_water_depletion(
        self,
        df: pd.DataFrame,
        field_capacity: float,
        permanent_wilting_point: float,
        root_zone_depth_cm: float,
    ) -> pd.DataFrame:
        """
        Compute soil water depletion metrics.

        Args:
            df: DataFrame with soil_moisture (VWC fraction)
            field_capacity: FC (VWC fraction)
            permanent_wilting_point: PWP (VWC fraction)
            root_zone_depth_cm: Root zone depth in cm

        Returns:
            DataFrame with added depletion columns

        Raises:
            ValueError: If soil_moisture column is missing or FC <= PWP
        """
        if "soil_moisture" not in df.columns:
            raise ValueError("soil_moisture column is required for soil water depletion calculation")
        if field_capacity <= permanent_wilting_point:
            raise ValueError("field_capacity must be greater than permanent_wilting_point")

        df = df.copy()

        ta = (field_capacity - permanent_wilting_point) * root_zone_depth_cm * 10
        df["taw_mm"] = ta

        depletion = (field_capacity - df["soil_moisture"]) * root_zone_depth_cm * 10
        df["depletion_mm"] = depletion.clip(lower=0)

        df["depletion_fraction"] = (
            (field_capacity - df["soil_moisture"])
            / (field_capacity - permanent_wilting_point)
        ).clip(0, 1)

        df["raw_mm"] = ta * 0.5

        return df

    def compute_rainfall_accumulation(
        self,
        df: pd.DataFrame,
        windows: List[int] = [1, 3, 7, 14],
    ) -> pd.DataFrame:
        """
        Compute rolling rainfall accumulation.

        Args:
            df: DataFrame with rainfall (mm) and timestamp
            windows: List of window sizes in days

        Returns:
            DataFrame with added rainfall accumulation columns
        """
        df = df.copy()

        if "rainfall" not in df.columns or "timestamp" not in df.columns:
            return df

        df = df.set_index("timestamp").sort_index()

        for window in windows:
            col_name = f"rainfall_{window}d_mm"
            df[col_name] = df["rainfall"].rolling(f"{window}D", min_periods=1).sum()

        return df.reset_index()

    def compute_soil_moisture_trend(
        self,
        df: pd.DataFrame,
        window: int = 24,
    ) -> pd.DataFrame:
        """
        Compute soil moisture trend (rate of change).

        Args:
            df: DataFrame with soil_moisture
            window: Rolling window in hours

        Returns:
            DataFrame with added sm_trend column
        """
        df = df.copy()

        if "soil_moisture" not in df.columns:
            return df

        df["sm_trend"] = df["soil_moisture"].rolling(window, min_periods=2).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) >= 2 else 0,
            raw=True,
        )

        return df

    def compute_growth_stage_features(
        self,
        df: pd.DataFrame,
        crop: str,
        sowing_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add crop growth stage features based on days after sowing.

        Args:
            df: DataFrame with timestamp
            crop: Crop name
            sowing_date: Sowing date string (YYYY-MM-DD)

        Returns:
            DataFrame with added growth stage features
        """
        df = df.copy()

        if "timestamp" not in df.columns or sowing_date is None:
            return df

        sowing = pd.Timestamp(sowing_date)
        df["days_after_sowing"] = (df["timestamp"] - sowing).dt.days

        if crop.lower() == "wheat":
            df["growth_stage"] = df["days_after_sowing"].apply(self._wheat_growth_stage)
        elif crop.lower() == "soybean":
            df["growth_stage"] = df["days_after_sowing"].apply(self._soybean_growth_stage)
        else:
            df["growth_stage"] = "unknown"

        return df

    def _wheat_growth_stage(self, das: int) -> str:
        """Map DAS to wheat growth stage."""
        if das < 0:
            return "pre_sowing"
        elif das <= 20:
            return "emergence"
        elif das <= 45:
            return "tillering"
        elif das <= 75:
            return "jointing"
        elif das <= 100:
            return "heading"
        elif das <= 125:
            return "grain_filling"
        else:
            return "maturity"

    def _soybean_growth_stage(self, das: int) -> str:
        """Map DAS to soybean growth stage."""
        if das < 0:
            return "pre_sowing"
        elif das <= 10:
            return "emergence"
        elif das <= 30:
            return "vegetative"
        elif das <= 55:
            return "flowering"
        elif das <= 85:
            return "pod_filling"
        else:
            return "maturity"

    def create_all_features(
        self,
        df: pd.DataFrame,
        field_capacity: Optional[float] = None,
        permanent_wilting_point: Optional[float] = None,
        root_zone_depth_cm: Optional[float] = None,
        latitude: Optional[float] = None,
        elevation: float = 0.0,
        crop: Optional[str] = None,
        sowing_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Create all available features.

        Args:
            df: Preprocessed sensor DataFrame
            field_capacity: FC (VWC fraction)
            permanent_wilting_point: PWP (VWC fraction)
            root_zone_depth_cm: Root zone depth in cm
            latitude: Site latitude for ET0
            elevation: Site elevation in meters
            crop: Crop name for growth stage
            sowing_date: Sowing date for growth stage

        Returns:
            DataFrame with all engineered features
        """
        df = df.copy()

        df = self.compute_vpd(df)

        if latitude is not None:
            df = self.compute_et0_hargreaves(df, latitude, elevation)

        if all(v is not None for v in [field_capacity, permanent_wilting_point, root_zone_depth_cm]):
            df = self.compute_soil_water_depletion(
                df, field_capacity, permanent_wilting_point, root_zone_depth_cm
            )

        df = self.compute_rainfall_accumulation(df)
        df = self.compute_soil_moisture_trend(df)

        if crop and sowing_date:
            df = self.compute_growth_stage_features(df, crop, sowing_date)

        return df


def engineer_features(
    df: pd.DataFrame,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function for feature engineering."""
    engineer = FeatureEngineer()
    return engineer.create_all_features(df, **kwargs)