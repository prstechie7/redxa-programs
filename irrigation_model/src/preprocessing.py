"""
Preprocessing module for sensor data.
Handles missing values, unit conversions, and data cleaning.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .config import (
    REQUIRED_SENSOR_COLUMNS,
    OPTIONAL_SENSOR_COLUMNS,
    SENSOR_DATA_PATH,
    DATA_PROCESSED_DIR,
    UNITS,
)
from .dataset_inspection import DatasetInspector


class SensorPreprocessor:
    """Preprocesses sensor CSV data for irrigation analysis."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or SENSOR_DATA_PATH
        self._df: Optional[pd.DataFrame] = None
        self._processed_df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load raw sensor data."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Sensor data not found: {self.data_path}")

        self._df = pd.read_csv(self.data_path)
        return self._df

    def validate_required_columns(self) -> List[str]:
        """Check which required columns are missing."""
        if self._df is None:
            self.load()

        missing = []
        for col in REQUIRED_SENSOR_COLUMNS:
            if col not in self._df.columns:
                missing.append(col)
        return missing

    def parse_timestamps(self) -> pd.DataFrame:
        """Parse timestamp column to datetime."""
        if self._df is None:
            self.load()

        df = self._df.copy()
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    def handle_missing_values(
        self,
        method: str = "interpolate",
        limit: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Handle missing values in sensor data.

        Args:
            method: 'interpolate', 'forward_fill', 'backward_fill', 'drop'
            limit: Maximum number of consecutive NaNs to fill

        Returns:
            DataFrame with missing values handled
        """
        if self._df is None:
            self.load()

        df = self._df.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns

        if method == "interpolate":
            use_time_method = (
                "timestamp" in df.columns
                and pd.api.types.is_datetime64_any_dtype(df["timestamp"])
            )
            if use_time_method:
                df = df.set_index("timestamp")
            df[numeric_cols] = df[numeric_cols].interpolate(
                method="time" if use_time_method else "linear",
                limit=limit,
                limit_direction="both",
            )
            if use_time_method:
                df = df.reset_index()
        elif method == "forward_fill":
            df[numeric_cols] = df[numeric_cols].ffill(limit=limit)
        elif method == "backward_fill":
            df[numeric_cols] = df[numeric_cols].bfill(limit=limit)
        elif method == "drop":
            df = df.dropna(subset=numeric_cols)
        else:
            raise ValueError(f"Unknown method: {method}")

        return df

    def validate_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip or flag values outside physical ranges.

        Returns:
            DataFrame with validated data
        """
        df = df.copy()

        range_bounds = {
            "soil_moisture": (0.0, 1.0),
            "soil_temperature": (-10.0, 60.0),
            "air_temperature": (-10.0, 55.0),
            "air_humidity": (0.0, 100.0),
            "rainfall": (0.0, 500.0),
            "solar_radiation": (0.0, 1500.0),
            "wind_speed": (0.0, 50.0),
        }

        for col, (min_val, max_val) in range_bounds.items():
            if col in df.columns:
                out_of_range = (df[col] < min_val) | (df[col] > max_val)
                if out_of_range.any():
                    df.loc[out_of_range, col] = np.nan

        return df

    def resample(
        self,
        df: pd.DataFrame,
        freq: str = "1H",
        agg_method: str = "mean",
    ) -> pd.DataFrame:
        """
        Resample time series to regular frequency.

        Args:
            freq: Pandas frequency string (e.g., '1H', '30T', '1D')
            agg_method: Aggregation method ('mean', 'sum', 'last')

        Returns:
            Resampled DataFrame
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must have timestamp column")

        df = df.set_index("timestamp")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        agg_dict = {col: agg_method for col in numeric_cols}

        if "rainfall" in df.columns:
            agg_dict["rainfall"] = "sum"

        resampled = df.resample(freq).agg(agg_dict)
        return resampled.reset_index()

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            df["hour"] = ts.dt.hour
            df["day_of_year"] = ts.dt.dayofyear
            df["month"] = ts.dt.month
            df["season"] = self._get_indian_season(ts)
        return df

    def _get_indian_season(self, timestamps: pd.Series) -> pd.Series:
        """Map months to Indian seasons."""
        def map_season(month):
            if month in [3, 4, 5]:
                return "Summer"
            elif month in [6, 7, 8, 9]:
                return "Kharif"
            elif month in [10, 11]:
                return "Post-Monsoon"
            else:
                return "Rabi"
        return timestamps.dt.month.apply(map_season)

    def process(
        self,
        handle_missing: str = "interpolate",
        resample_freq: Optional[str] = None,
        validate_ranges: bool = True,
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.

        Args:
            handle_missing: Method for missing values
            resample_freq: Optional resampling frequency
            validate_ranges: Whether to validate physical ranges

        Returns:
            Processed DataFrame
        """
        missing = self.validate_required_columns()
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = self.parse_timestamps()
        df = self.handle_missing_values(method=handle_missing)

        if validate_ranges:
            df = self.validate_ranges(df)

        if resample_freq:
            df = self.resample(df, freq=resample_freq)

        df = self.add_time_features(df)

        self._processed_df = df
        return df

    def save_processed(self, output_path: Optional[Path] = None) -> Path:
        """Save processed data to CSV."""
        if self._processed_df is None:
            raise ValueError("No processed data. Run process() first.")

        if output_path is None:
            output_path = DATA_PROCESSED_DIR / "sensor_data_processed.csv"

        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._processed_df.to_csv(output_path, index=False)
        return output_path


def preprocess_sensor_data(
    data_path: Optional[Path] = None,
    handle_missing: str = "interpolate",
    resample_freq: Optional[str] = None,
    validate_ranges: bool = True,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Convenience function for full preprocessing pipeline."""
    preprocessor = SensorPreprocessor(data_path)
    df = preprocessor.process(
        handle_missing=handle_missing,
        resample_freq=resample_freq,
        validate_ranges=validate_ranges,
    )
    if output_path:
        preprocessor.save_processed(output_path)
    return df