"""
Threshold database management for irrigation module.
Provides CRUD operations for the research-backed threshold database.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from .config import (
    THRESHOLD_DATABASE_PATH,
    VALID_EVIDENCE_LEVELS,
    VALID_THRESHOLD_TYPES,
)


class ThresholdDatabase:
    """Manages the irrigation threshold database."""

    REQUIRED_COLUMNS = [
        "crop",
        "state",
        "region",
        "agro_climatic_zone",
        "season",
        "soil_type",
        "growth_stage",
        "parameter_type",
        "parameter_name",
        "parameter_value",
        "parameter_unit",
        "source",
        "source_url",
        "doi",
        "evidence_level",
        "verification_status",
        "notes",
        "actual_study_location",
        "actual_soil_context",
        "actual_crop_system",
        "parameter_applicability",
        "proxy_for",
        "calibration_required",
    ]

    VALID_VERIFICATION_STATUSES = [
        "VERIFIED_EXACT",
        "VERIFIED_CONTEXTUAL",
        "DERIVED",
        "NOT_VERIFIED",
        "REQUIRES_FIELD_CALIBRATION",
        "PROXY_INDIAN",
        "FIELD_CALIBRATED",
        "REQUIRES_CALIBRATION",
    ]

    VALID_PARAMETER_TYPES = [
        "seasonal_crop_water_requirement",
        "soil_water_trigger",
        "soil_hydraulic_parameter",
        "crop_coefficient",
        "root_zone_parameter",
        "et0_methodology",
        "available_soil_moisture_depletion",
        "iwcpe_ratio",
    ]

    def __init__(self, database_path: Optional[Path] = None):
        self.database_path = database_path or THRESHOLD_DATABASE_PATH
        self._df: Optional[pd.DataFrame] = None
        self._load()

    def _load(self) -> None:
        """Load database from CSV."""
        if self.database_path.exists():
            self._df = pd.read_csv(self.database_path)
            for col in self.REQUIRED_COLUMNS:
                if col not in self._df.columns:
                    self._df[col] = None
        else:
            self._df = pd.DataFrame(columns=self.REQUIRED_COLUMNS)

    def _save(self) -> None:
        """Save database to CSV."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(self.database_path, index=False)

    def validate_record(self, record: Dict[str, Any]) -> List[str]:
        """Validate a threshold record. Returns list of errors."""
        errors = []

        if not record.get("crop"):
            errors.append("crop is required")
        if not record.get("parameter_type"):
            errors.append("parameter_type is required")
        elif record["parameter_type"] not in self.VALID_PARAMETER_TYPES:
            errors.append(f"Invalid parameter_type: {record['parameter_type']}")

        if record.get("evidence_level"):
            if record["evidence_level"] not in VALID_EVIDENCE_LEVELS:
                errors.append(f"Invalid evidence_level: {record['evidence_level']}")

        if record.get("verification_status"):
            if record["verification_status"] not in self.VALID_VERIFICATION_STATUSES:
                errors.append(f"Invalid verification_status: {record['verification_status']}")

        if record.get("parameter_value") is not None:
            try:
                float(record["parameter_value"])
            except (ValueError, TypeError):
                errors.append("parameter_value must be numeric")

        return errors

    def add_record(self, record: Dict[str, Any]) -> bool:
        """
        Add a new threshold record.

        Args:
            record: Dictionary with threshold data

        Returns:
            True if added successfully

        Raises:
            ValueError: If validation fails
        """
        errors = self.validate_record(record)
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")

        full_record = {col: record.get(col) for col in self.REQUIRED_COLUMNS}
        self._df = pd.concat([self._df, pd.DataFrame([full_record])], ignore_index=True)
        self._save()
        return True

    def update_record(self, index: int, record: Dict[str, Any]) -> bool:
        """Update an existing record by index."""
        if index < 0 or index >= len(self._df):
            raise IndexError("Record index out of range")

        errors = self.validate_record(record)
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")

        for col in self.REQUIRED_COLUMNS:
            if col in record:
                self._df.at[index, col] = record[col]
        self._save()
        return True

    def delete_record(self, index: int) -> bool:
        """Delete a record by index."""
        if index < 0 or index >= len(self._df):
            raise IndexError("Record index out of range")
        self._df = self._df.drop(index).reset_index(drop=True)
        self._save()
        return True

    def get_all(self) -> pd.DataFrame:
        """Get all records."""
        return self._df.copy()

    def query(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        parameter_type: Optional[str] = None,
        evidence_level: Optional[str] = None,
        verification_status: Optional[str] = None,
        growth_stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """Query database with filters."""
        if self._df is None or self._df.empty:
            return pd.DataFrame(columns=self.REQUIRED_COLUMNS)

        mask = pd.Series([True] * len(self._df), index=self._df.index)

        if crop:
            mask &= self._df["crop"].str.lower() == crop.lower()
        if state:
            mask &= self._df["state"].str.lower() == state.lower()
        if region:
            mask &= self._df["region"].str.lower() == region.lower()
        if soil_type:
            mask &= self._df["soil_type"].str.lower() == soil_type.lower()
        if season:
            mask &= self._df["season"].str.lower() == season.lower()
        if parameter_type:
            mask &= self._df["parameter_type"] == parameter_type
        if evidence_level:
            mask &= self._df["evidence_level"] == evidence_level
        if verification_status:
            mask &= self._df["verification_status"] == verification_status
        if growth_stage:
            mask &= (
                (self._df["growth_stage"].str.lower() == growth_stage.lower())
                | (self._df["growth_stage"].str.lower() == "all")
                | (self._df["growth_stage"].isna())
                | (self._df["growth_stage"] == "")
            )

        return self._df[mask].copy()

    def get_seasonal_requirements(self) -> pd.DataFrame:
        """Get all seasonal crop water requirement records."""
        return self.query(parameter_type="seasonal_crop_water_requirement")

    def get_irrigation_triggers(self) -> pd.DataFrame:
        """Get all irrigation trigger thresholds."""
        return self.query(parameter_type="soil_water_trigger")

    def get_verified_parameters(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        parameter_type: Optional[str] = None,
        growth_stage: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get only VERIFIED_EXACT or VERIFIED_CONTEXTUAL parameters."""
        return self.query(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            parameter_type=parameter_type,
            verification_status="VERIFIED_EXACT",
            growth_stage=growth_stage,
        ).pipe(lambda df: pd.concat([
            df,
            self.query(
                crop=crop,
                state=state,
                region=region,
                soil_type=soil_type,
                season=season,
                parameter_type=parameter_type,
                verification_status="VERIFIED_CONTEXTUAL",
                growth_stage=growth_stage,
            )
        ]).drop_duplicates()) if not df.empty else df

    def get_parameters_requiring_calibration(
        self,
        crop: Optional[str] = None,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get parameters that require field calibration."""
        return self.query(
            crop=crop,
            state=state,
            region=region,
            soil_type=soil_type,
            season=season,
            verification_status="REQUIRES_FIELD_CALIBRATION",
        )

    def export_to_csv(self, output_path: Path) -> None:
        """Export database to CSV."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(output_path, index=False)

    def import_from_csv(self, input_path: Path, validate: bool = True) -> int:
        """Import records from CSV."""
        new_df = pd.read_csv(input_path)
        for col in self.REQUIRED_COLUMNS:
            if col not in new_df.columns:
                new_df[col] = None

        added = 0
        for _, row in new_df.iterrows():
            record = row.to_dict()
            if validate:
                errors = self.validate_record(record)
                if errors:
                    continue
            self.add_record(record)
            added += 1
        return added


_database_instance: Optional[ThresholdDatabase] = None


def get_database() -> ThresholdDatabase:
    """Get singleton instance of ThresholdDatabase."""
    global _database_instance
    if _database_instance is None:
        _database_instance = ThresholdDatabase()
    return _database_instance