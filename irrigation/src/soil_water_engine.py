"""
Soil-Water Engine - Core agronomic decision engine for irrigation scheduling.

Implements research-backed soil water balance calculations:
- TAW = (FC - PWP) * root_zone_depth
- RAW = MAD * TAW
- depletion = FC - current_soil_moisture
- depletion_fraction = depletion / (FC - PWP)

This is an AGRONOMIC decision engine, NOT the final ML model.
"""
from dataclasses import dataclass
from typing import Optional, Literal
import numpy as np

from .config import VALID_STATUS


@dataclass
class SoilWaterParams:
    """Soil water parameters for a specific crop/soil combination."""
    field_capacity: float
    permanent_wilting_point: float
    root_zone_depth_cm: float
    mad: float = 0.5

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Validate soil water parameters."""
        if not (0 <= self.permanent_wilting_point < self.field_capacity <= 1):
            raise ValueError(
                f"Invalid FC/PWP: PWP={self.permanent_wilting_point}, FC={self.field_capacity}. "
                f"Must satisfy 0 <= PWP < FC <= 1"
            )
        if self.root_zone_depth_cm <= 0:
            raise ValueError(
                f"root_zone_depth_cm must be positive, got {self.root_zone_depth_cm}"
            )
        if not (0 < self.mad <= 1):
            raise ValueError(f"MAD must be in (0, 1], got {self.mad}")


@dataclass
class SoilWaterResult:
    """Result of soil water balance calculation."""
    total_available_water_mm: float
    readily_available_water_mm: float
    current_depletion_mm: float
    depletion_fraction: float
    irrigation_triggered: bool
    status: Literal["IRRIGATION_REQUIRED", "NO_IRRIGATION_REQUIRED"]
    current_soil_moisture: float
    field_capacity: float
    permanent_wilting_point: float
    root_zone_depth_cm: float
    mad: float


class SoilWaterEngine:
    """
    Core agronomic decision engine for irrigation scheduling.

    Calculates soil water balance and determines irrigation need
    based on research-backed thresholds (MAD, depletion fraction).

    IMPORTANT: This is an agronomic decision engine, NOT the final ML model.
    """

    def __init__(self, params: SoilWaterParams):
        """
        Initialize engine with soil water parameters.

        Args:
            params: SoilWaterParams with FC, PWP, root_zone_depth_cm, MAD
        """
        self.params = params
        self._calculate_constants()

    def _calculate_constants(self) -> None:
        """Pre-calculate TAW and RAW."""
        self.taw_mm = (
            (self.params.field_capacity - self.params.permanent_wilting_point)
            * self.params.root_zone_depth_cm
            * 10.0
        )
        self.raw_mm = self.params.mad * self.taw_mm

    def calculate(
        self,
        current_soil_moisture: float,
        mad_override: Optional[float] = None,
    ) -> SoilWaterResult:
        """
        Calculate soil water status and irrigation trigger.

        Args:
            current_soil_moisture: Current volumetric water content (fraction, 0-1)
            mad_override: Optional MAD override for specific growth stage

        Returns:
            SoilWaterResult with all calculated values and irrigation decision
        """
        self._validate_moisture(current_soil_moisture)

        mad = mad_override if mad_override is not None else self.params.mad
        if not (0 < mad <= 1):
            raise ValueError(f"MAD must be in (0, 1], got {mad}")

        raw_mm = mad * self.taw_mm

        depletion_mm = (
            (self.params.field_capacity - current_soil_moisture)
            * self.params.root_zone_depth_cm
            * 10.0
        )
        depletion_mm = max(0.0, depletion_mm)

        depletion_fraction = depletion_mm / self.taw_mm if self.taw_mm > 0 else 0.0
        depletion_fraction = min(1.0, max(0.0, depletion_fraction))

        irrigation_triggered = depletion_mm >= raw_mm
        status: Literal["IRRIGATION_REQUIRED", "NO_IRRIGATION_REQUIRED"] = (
            "IRRIGATION_REQUIRED" if irrigation_triggered else "NO_IRRIGATION_REQUIRED"
        )

        return SoilWaterResult(
            total_available_water_mm=self.taw_mm,
            readily_available_water_mm=raw_mm,
            current_depletion_mm=depletion_mm,
            depletion_fraction=depletion_fraction,
            irrigation_triggered=irrigation_triggered,
            status=status,
            current_soil_moisture=current_soil_moisture,
            field_capacity=self.params.field_capacity,
            permanent_wilting_point=self.params.permanent_wilting_point,
            root_zone_depth_cm=self.params.root_zone_depth_cm,
            mad=mad,
        )

    def _validate_moisture(self, moisture: float) -> None:
        """Validate soil moisture value."""
        if not (0 <= moisture <= 1):
            raise ValueError(
                f"soil_moisture must be in [0, 1] (VWC fraction), got {moisture}"
            )

    def get_irrigation_depth_mm(
        self,
        current_soil_moisture: float,
        target_fraction: float = 0.9,
    ) -> float:
        """
        Calculate recommended irrigation depth to reach target moisture.

        Args:
            current_soil_moisture: Current VWC fraction
            target_fraction: Target fraction of TAW to refill (default 0.9 = near FC)

        Returns:
            Irrigation depth in mm
        """
        self._validate_moisture(current_soil_moisture)

        target_moisture = (
            self.params.permanent_wilting_point
            + target_fraction * (self.params.field_capacity - self.params.permanent_wilting_point)
        )
        target_moisture = min(target_moisture, self.params.field_capacity)

        deficit_mm = (
            (target_moisture - current_soil_moisture)
            * self.params.root_zone_depth_cm
            * 10.0
        )

        return max(0.0, deficit_mm)

    def get_depletion_at_mad(self, mad: Optional[float] = None) -> float:
        """Get depletion (mm) at specified MAD."""
        mad = mad if mad is not None else self.params.mad
        return mad * self.taw_mm

    def get_moisture_at_mad(self, mad: Optional[float] = None) -> float:
        """Get soil moisture (VWC fraction) at specified MAD."""
        mad = mad if mad is not None else self.params.mad
        return self.params.field_capacity - mad * (
            self.params.field_capacity - self.params.permanent_wilting_point
        )


def calculate_taw(
    field_capacity: float,
    permanent_wilting_point: float,
    root_zone_depth_cm: float,
) -> float:
    """
    Calculate Total Available Water (TAW) in mm.

    TAW = (FC - PWP) * root_zone_depth_cm * 10

    Args:
        field_capacity: FC (VWC fraction, 0-1)
        permanent_wilting_point: PWP (VWC fraction, 0-1)
        root_zone_depth_cm: Root zone depth in cm

    Returns:
        TAW in mm
    """
    if not (0 <= permanent_wilting_point < field_capacity <= 1):
        raise ValueError("Must satisfy 0 <= PWP < FC <= 1")
    if root_zone_depth_cm <= 0:
        raise ValueError("root_zone_depth_cm must be positive")

    return (field_capacity - permanent_wilting_point) * root_zone_depth_cm * 10.0


def calculate_raw(
    taw_mm: float,
    mad: float,
) -> float:
    """
    Calculate Readily Available Water (RAW) in mm.

    RAW = MAD * TAW

    Args:
        taw_mm: Total available water in mm
        mad: Management allowed depletion (fraction, 0-1)

    Returns:
        RAW in mm
    """
    if not (0 < mad <= 1):
        raise ValueError("MAD must be in (0, 1]")
    if taw_mm < 0:
        raise ValueError("TAW must be non-negative")

    return mad * taw_mm


def calculate_depletion(
    field_capacity: float,
    current_soil_moisture: float,
    root_zone_depth_cm: float,
) -> float:
    """
    Calculate current soil water depletion in mm.

    depletion = (FC - current_soil_moisture) * root_zone_depth_cm * 10

    Args:
        field_capacity: FC (VWC fraction)
        current_soil_moisture: Current VWC fraction
        root_zone_depth_cm: Root zone depth in cm

    Returns:
        Depletion in mm
    """
    if not (0 <= current_soil_moisture <= 1):
        raise ValueError("soil_moisture must be in [0, 1]")
    if current_soil_moisture > field_capacity:
        return 0.0

    return (field_capacity - current_soil_moisture) * root_zone_depth_cm * 10.0


def calculate_depletion_fraction(
    field_capacity: float,
    permanent_wilting_point: float,
    current_soil_moisture: float,
) -> float:
    """
    Calculate depletion fraction (0-1).

    depletion_fraction = (FC - current) / (FC - PWP)

    Args:
        field_capacity: FC (VWC fraction)
        permanent_wilting_point: PWP (VWC fraction)
        current_soil_moisture: Current VWC fraction

    Returns:
        Depletion fraction (0-1)
    """
    if not (0 <= permanent_wilting_point < field_capacity <= 1):
        raise ValueError("Must satisfy 0 <= PWP < FC <= 1")
    if not (0 <= current_soil_moisture <= 1):
        raise ValueError("soil_moisture must be in [0, 1]")

    if field_capacity == permanent_wilting_point:
        return 0.0

    fraction = (field_capacity - current_soil_moisture) / (
        field_capacity - permanent_wilting_point
    )
    return min(1.0, max(0.0, fraction))


def vwc_percent_to_fraction(vwc_percent: float) -> float:
    """Convert VWC percent to fraction. 30% -> 0.30"""
    return vwc_percent / 100.0


def vwc_fraction_to_percent(vwc_fraction: float) -> float:
    """Convert VWC fraction to percent. 0.30 -> 30%"""
    return vwc_fraction * 100.0


def kpa_to_cbar(kpa: float) -> float:
    """Convert kPa to centibar. 1 kPa = 1 cbar"""
    return kpa


def cbar_to_kpa(cbar: float) -> float:
    """Convert centibar to kPa. 1 cbar = 1 kPa"""
    return cbar