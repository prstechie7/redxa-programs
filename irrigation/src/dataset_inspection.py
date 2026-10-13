"""
Dataset inspection utilities for sensor data.
Provides validation and quality checks for incoming CSV data.
"""
from pathlib import Path
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np

from .config import (
    REQUIRED_SENSOR_COLUMNS,
    OPTIONAL_SENSOR_COLUMNS,
    SENSOR_DATA_PATH,
    UNITS,
)


class DatasetInspector:
    """Inspects and validates sensor CSV data."""

    def __init__(self, data_path: Optional[Path] = None):
        self.data_path = data_path or SENSOR_DATA_PATH
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load sensor data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Sensor data not found: {self.data_path}")

        self._df = pd.read_csv(self.data_path)
        return self._df

    def inspect(self) -> Dict[str, Any]:
        """
        Perform comprehensive inspection of the dataset.

        Returns:
            Dictionary with inspection results
        """
        if self._df is None:
            self.load()

        df = self._df
        results = {
            "file_path": str(self.data_path),
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
            "date_range": None,
            "numeric_stats": {},
            "required_columns_present": True,
            "missing_required": [],
            "optional_columns_present": [],
            "warnings": [],
        }

        for col in REQUIRED_SENSOR_COLUMNS:
            if col not in df.columns:
                results["required_columns_present"] = False
                results["missing_required"].append(col)

        for col in OPTIONAL_SENSOR_COLUMNS:
            if col in df.columns:
                results["optional_columns_present"].append(col)

        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                results["date_range"] = {
                    "start": df["timestamp"].min().isoformat(),
                    "end": df["timestamp"].max().isoformat(),
                    "frequency": self._infer_frequency(df["timestamp"]),
                }
            except Exception as e:
                results["warnings"].append(f"Timestamp parsing failed: {e}")

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            results["numeric_stats"][col] = {
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
            }

        if "soil_moisture" in df.columns:
            invalid_moisture = df[
                (df["soil_moisture"] < 0) | (df["soil_moisture"] > 1)
            ]
            if len(invalid_moisture) > 0:
                results["warnings"].append(
                    f"Found {len(invalid_moisture)} rows with soil_moisture "
                    f"outside valid range [0, 1] (VWC fraction)"
                )

        return results

    def _infer_frequency(self, timestamps: pd.Series) -> Optional[str]:
        """Infer timestamp frequency."""
        if len(timestamps) < 2:
            return None
        diffs = timestamps.diff().dropna()
        if len(diffs) == 0:
            return None
        mode_diff = diffs.mode()
        if len(mode_diff) > 0:
            return str(mode_diff.iloc[0])
        return None

    def validate_ranges(self) -> Dict[str, List[int]]:
        """
        Validate sensor values are within expected physical ranges.

        Returns:
            Dict mapping column names to list of row indices with invalid values
        """
        if self._df is None:
            self.load()

        df = self._df
        invalid = {}

        range_checks = {
            "soil_moisture": (0, 1),
            "soil_temperature": (-10, 60),
            "air_temperature": (-10, 55),
            "air_humidity": (0, 100),
            "rainfall": (0, 500),
            "solar_radiation": (0, 1500),
            "wind_speed": (0, 50),
        }

        for col, (min_val, max_val) in range_checks.items():
            if col in df.columns:
                mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_indices = df[mask].index.tolist()
                if invalid_indices:
                    invalid[col] = invalid_indices

        return invalid

    def print_summary(self) -> None:
        """Print human-readable summary."""
        results = self.inspect()
        print(f"\n=== Sensor Data Inspection ===")
        print(f"File: {results['file_path']}")
        print(f"Shape: {results['shape'][0]} rows x {results['shape'][1]} columns")
        print(f"Columns: {', '.join(results['columns'])}")

        if results["date_range"]:
            dr = results["date_range"]
            print(f"Date Range: {dr['start']} to {dr['end']}")
            print(f"Frequency: {dr['frequency']}")

        print(f"\nRequired columns present: {results['required_columns_present']}")
        if results["missing_required"]:
            print(f"Missing required: {results['missing_required']}")

        print(f"Optional columns present: {results['optional_columns_present']}")

        print(f"\nMissing values:")
        for col, count in results["missing_values"].items():
            pct = results["missing_percentage"][col]
            print(f"  {col}: {count} ({pct:.1f}%)")

        if results["warnings"]:
            print(f"\nWarnings:")
            for w in results["warnings"]:
                print(f"  - {w}")

        invalid = self.validate_ranges()
        if invalid:
            print(f"\nRange validation failures:")
            for col, indices in invalid.items():
                print(f"  {col}: {len(indices)} invalid rows")


def inspect_sensor_data(data_path: Optional[Path] = None) -> Dict[str, Any]:
    """Convenience function to inspect sensor data."""
    inspector = DatasetInspector(data_path)
    return inspector.inspect()