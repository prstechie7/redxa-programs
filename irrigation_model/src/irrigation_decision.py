"""
Irrigation Decision Engine - Deterministic soil-water-based irrigation decisions.

Orchestrates:
- Parameter lookup from threshold database
- Soil water balance calculations
- Threshold selection with evidence tracking
- Irrigation decision with explanation

This is an AGRONOMIC decision engine, NOT the final ML model.
"""
from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List
from enum import Enum
import numpy as np

from .config import VALID_STATUS
from .threshold_config import get_threshold_config, get_crop_profile
from .soil_water_engine import (
    SoilWaterEngine,
    SoilWaterParams,
    SoilWaterResult,
    calculate_taw,
    calculate_raw,
    calculate_depletion,
    calculate_depletion_fraction,
    vwc_percent_to_fraction,
    vwc_fraction_to_percent,
)
from .threshold_database import ThresholdDatabase


class ThresholdType(Enum):
    """Types of irrigation scheduling parameters."""
    MAD = "maximum_allowable_depletion"           # Locally validated operational MAD/p
    ASMD = "available_soil_moisture_depletion"    # Research ASMD values (contextual)
    IWCPE = "iwcpe_ratio"                         # IW:CPE irrigation scheduling ratio
    SOIL_MATRIC = "soil_matric_potential"         # Tensiometer-based threshold


class EvidenceLevel(Enum):
    """Evidence level for threshold used in decision."""
    RESEARCH_BACKED = "RESEARCH_BACKED"           # VERIFIED_EXACT local
    RESEARCH_CONTEXTUAL = "RESEARCH_CONTEXTUAL"   # VERIFIED_CONTEXTUAL local
    INDIAN_CONTEXTUAL = "INDIAN_CONTEXTUAL"       # Other Indian research
    PROXY_INDIAN = "PROXY_INDIAN"                 # Proxy from other region
    USER_SUPPLIED = "USER_SUPPLIED"               # User provided
    REQUIRES_CALIBRATION = "REQUIRES_CALIBRATION" # No local value available


@dataclass
class ThresholdReference:
    """A research threshold reference with full provenance."""
    parameter_type: str
    parameter_name: str
    parameter_value: float
    parameter_unit: str
    evidence_level: EvidenceLevel
    verification_status: str
    source: str
    notes: str
    actual_study_location: str
    actual_soil_context: str
    actual_crop_system: str
    is_operational: bool = False  # Can be used as operational trigger


@dataclass
class IrrigationDecision:
    """Complete irrigation decision with explanation and evidence."""
    # Decision
    irrigation_required: bool
    status: Literal["IRRIGATION_REQUIRED", "NO_IRRIGATION_REQUIRED"]

    # Soil water status
    current_soil_moisture: float
    field_capacity: float
    permanent_wilting_point: float
    root_zone_depth_cm: float
    depletion_mm: float
    depletion_fraction: float
    taw_mm: float
    raw_mm: float

    @property
    def current_depletion_mm(self) -> float:
        """Alias for depletion_mm for backward compatibility."""
        return self.depletion_mm

    # Threshold used
    threshold_used: Optional[float] = None
    threshold_type: Optional[ThresholdType] = None
    threshold_parameter_type: Optional[str] = None
    threshold_evidence_level: Optional[EvidenceLevel] = None
    threshold_verification_status: Optional[str] = None
    threshold_source: Optional[str] = None
    threshold_notes: Optional[str] = None
    threshold_actual_study_location: Optional[str] = None
    threshold_actual_soil_context: Optional[str] = None
    threshold_actual_crop_system: Optional[str] = None

    # Growth stage
    growth_stage: Optional[str] = None
    kc_value: Optional[float] = None

    # Context
    crop: str = ""
    state: str = ""
    region: str = ""
    soil_type: str = ""

    # Explanation
    explanation: str = ""
    warnings: List[str] = field(default_factory=list)
    calibration_required: bool = False
    calibration_notes: str = ""

    # Reference thresholds (contextual)
    reference_thresholds: List[ThresholdReference] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "irrigation_required": self.irrigation_required,
            "status": self.status,
            "current_soil_moisture": self.current_soil_moisture,
            "field_capacity": self.field_capacity,
            "permanent_wilting_point": self.permanent_wilting_point,
            "root_zone_depth_cm": self.root_zone_depth_cm,
            "depletion_mm": self.depletion_mm,
            "depletion_fraction": self.depletion_fraction,
            "taw_mm": self.taw_mm,
            "raw_mm": self.raw_mm,
            "threshold_used": self.threshold_used,
            "threshold_type": self.threshold_type.value if self.threshold_type else None,
            "threshold_parameter_type": self.threshold_parameter_type,
            "threshold_evidence_level": self.threshold_evidence_level.value if self.threshold_evidence_level else None,
            "threshold_verification_status": self.threshold_verification_status,
            "threshold_source": self.threshold_source,
            "threshold_notes": self.threshold_notes,
            "threshold_actual_study_location": self.threshold_actual_study_location,
            "threshold_actual_soil_context": self.threshold_actual_soil_context,
            "threshold_actual_crop_system": self.threshold_actual_crop_system,
            "growth_stage": self.growth_stage,
            "kc_value": self.kc_value,
            "crop": self.crop,
            "state": self.state,
            "region": self.region,
            "soil_type": self.soil_type,
            "explanation": self.explanation,
            "warnings": self.warnings,
            "calibration_required": self.calibration_required,
            "calibration_notes": self.calibration_notes,
            "reference_thresholds": [
                {
                    "parameter_type": r.parameter_type,
                    "parameter_name": r.parameter_name,
                    "parameter_value": r.parameter_value,
                    "parameter_unit": r.parameter_unit,
                    "evidence_level": r.evidence_level.value,
                    "verification_status": r.verification_status,
                    "source": r.source,
                    "notes": r.notes,
                    "actual_study_location": r.actual_study_location,
                    "actual_soil_context": r.actual_soil_context,
                    "actual_crop_system": r.actual_crop_system,
                    "is_operational": r.is_operational,
                }
                for r in self.reference_thresholds
            ],
        }


class IrrigationDecisionEngine:
    """
    Deterministic irrigation decision engine.

    Uses research-backed thresholds from the database to make irrigation
    decisions based on soil-water balance calculations.

    IMPORTANT: Does NOT auto-convert ASMD to MAD or IW:CPE to ASMD.
    """

    def __init__(
        self,
        crop: str,
        state: Optional[str] = None,
        region: Optional[str] = None,
        soil_type: Optional[str] = None,
        season: Optional[str] = None,
        growth_stage: Optional[str] = None,
        current_soil_moisture: Optional[float] = None,
        current_soil_moisture_unit: str = "fraction",  # "fraction" or "percent"
        user_mad: Optional[float] = None,
        user_fc: Optional[float] = None,
        user_pwp: Optional[float] = None,
        user_root_zone_depth_cm: Optional[float] = None,
    ):
        """
        Initialize the irrigation decision engine.

        Args:
            crop: Crop name (e.g., "soybean", "wheat")
            state: Indian state
            region: Region within state
            soil_type: Soil type (e.g., "Vertisol", "Alluvial")
            season: Growing season (e.g., "Kharif", "Rabi")
            growth_stage: Crop growth stage (e.g., "initial", "development", "mid", "late")
            current_soil_moisture: Current soil moisture (VWC fraction or percent)
            current_soil_moisture_unit: "fraction" (0-1) or "percent" (0-100)
            user_mad: User-supplied MAD override (fraction)
            user_fc: User-supplied FC override (fraction)
            user_pwp: User-supplied PWP override (fraction)
            user_root_zone_depth_cm: User-supplied root zone depth (cm)
        """
        self.crop = crop
        self.state = state
        self.region = region
        self.soil_type = soil_type
        self.season = season
        self.growth_stage = growth_stage
        self.user_mad = user_mad
        self.user_fc = user_fc
        self.user_pwp = user_pwp
        self.user_root_zone_depth_cm = user_root_zone_depth_cm

        # Handle moisture input unit
        if current_soil_moisture is not None:
            if current_soil_moisture_unit == "percent":
                self.current_soil_moisture = vwc_percent_to_fraction(current_soil_moisture)
            else:
                self.current_soil_moisture = current_soil_moisture
        else:
            self.current_soil_moisture = None

        self.config = get_threshold_config()
        self.db = ThresholdDatabase()

        # Will be populated during decision
        self._soil_params: Optional[SoilWaterParams] = None
        self._engine: Optional[SoilWaterEngine] = None
        self._result: Optional[SoilWaterResult] = None
        self._threshold_reference: Optional[ThresholdReference] = None
        self._reference_thresholds: List[ThresholdReference] = []
        self._kc_value: Optional[float] = None
        self._calibration_required = False
        self._calibration_notes = ""
        self._warnings: List[str] = []

    def _load_soil_hydraulic_params(self) -> Dict[str, Optional[float]]:
        """Load FC, PWP from database (verified only)."""
        return self.config.get_soil_hydraulic_parameters(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            include_unverified=False,
        )

    def _load_root_zone_depth(self) -> Optional[float]:
        """Load root zone depth from database (verified only)."""
        return self.config.get_root_zone_depth(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            include_unverified=False,
        )

    def _load_kc_value(self) -> Optional[float]:
        """Load Kc value for current growth stage from database."""
        if not self.growth_stage:
            return None

        kc_dict = self.config.get_crop_coefficients(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            include_unverified=False,
        )

        stage_key = f"kc_{self.growth_stage.lower()}"
        return kc_dict.get(stage_key)

    def _load_local_mad(self) -> Optional[Dict[str, Any]]:
        """Load locally validated MAD/p from database."""
        return self.config.get_soil_moisture_threshold(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            growth_stage=self.growth_stage,
            parameter_name="maximum_allowable_depletion",
            include_unverified=False,
        )

    def _load_all_contextual_thresholds(self) -> List[ThresholdReference]:
        """Load ALL available thresholds for context (ASMD, IW:CPE, etc.)."""
        references = []

        # Load all parameters for this crop/context
        profile = self.config.get_crop_profile(
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            season=self.season,
            growth_stage=self.growth_stage,
            include_unverified=True,
        )

        if profile.empty:
            return references

        for _, row in profile.iterrows():
            param_type = row.get("parameter_type", "")
            param_name = row.get("parameter_name", "")
            verification = row.get("verification_status", "")
            evidence = row.get("evidence_level", "")

            # Skip non-threshold parameters
            if param_type not in ["soil_water_trigger", "available_soil_moisture_depletion", "iwcpe_ratio"]:
                continue

            # Map verification status to evidence level
            if verification == "VERIFIED_EXACT":
                evidence_level = EvidenceLevel.RESEARCH_BACKED
                is_operational = True
            elif verification == "VERIFIED_CONTEXTUAL":
                evidence_level = EvidenceLevel.RESEARCH_CONTEXTUAL
                is_operational = True
            elif verification in ["DIRECT_INDIAN_CONTEXTUAL", "VERIFIED_CONTEXTUAL"]:
                evidence_level = EvidenceLevel.INDIAN_CONTEXTUAL
                is_operational = False
            elif evidence == "PROXY_INDIAN":
                evidence_level = EvidenceLevel.PROXY_INDIAN
                is_operational = False
            elif verification in ["REQUIRES_FIELD_CALIBRATION", "REQUIRES_CALIBRATION"]:
                evidence_level = EvidenceLevel.REQUIRES_CALIBRATION
                is_operational = False
            else:
                evidence_level = EvidenceLevel.REQUIRES_CALIBRATION
                is_operational = False

            # Determine threshold type
            if param_name == "maximum_allowable_depletion":
                ttype = ThresholdType.MAD
            elif param_name == "available_soil_moisture_depletion":
                ttype = ThresholdType.ASMD
            elif param_name == "iwcpe_ratio":
                ttype = ThresholdType.IWCPE
            elif param_name == "soil_matric_potential":
                ttype = ThresholdType.SOIL_MATRIC
            else:
                continue

            ref = ThresholdReference(
                parameter_type=param_type,
                parameter_name=param_name,
                parameter_value=float(row.get("parameter_value", 0)) if pd.notna(row.get("parameter_value")) else 0.0,
                parameter_unit=str(row.get("parameter_unit", "")),
                evidence_level=evidence_level,
                verification_status=str(verification),
                source=str(row.get("source", "")),
                notes=str(row.get("notes", "")),
                actual_study_location=str(row.get("actual_study_location", "")),
                actual_soil_context=str(row.get("actual_soil_context", "")),
                actual_crop_system=str(row.get("actual_crop_system", "")),
                is_operational=is_operational,
            )
            references.append(ref)

        return references

    def _validate_inputs(self) -> None:
        """Validate all inputs before calculation."""
        if self.current_soil_moisture is None:
            raise ValueError("current_soil_moisture is required")

        if not (0 <= self.current_soil_moisture <= 1):
            raise ValueError(f"current_soil_moisture must be in [0, 1] (fraction), got {self.current_soil_moisture}")

        if self.user_fc is not None and self.user_pwp is not None:
            if not (0 <= self.user_pwp < self.user_fc <= 1):
                raise ValueError(f"Invalid FC/PWP: PWP={self.user_pwp}, FC={self.user_fc}. Must satisfy 0 <= PWP < FC <= 1")

        if self.user_root_zone_depth_cm is not None and self.user_root_zone_depth_cm <= 0:
            raise ValueError(f"root_zone_depth_cm must be positive, got {self.user_root_zone_depth_cm}")

        if self.user_mad is not None and not (0 < self.user_mad <= 1):
            raise ValueError(f"MAD must be in (0, 1], got {self.user_mad}")

        # Validate moisture against FC/PWP if available (only for database-loaded params)
        fc = self.user_fc
        pwp = self.user_pwp
        fc_from_db = False
        pwp_from_db = False
        if fc is None or pwp is None:
            hydraulic = self._load_soil_hydraulic_params()
            fc = fc or hydraulic.get("field_capacity")
            pwp = pwp or hydraulic.get("permanent_wilting_point")
            fc_from_db = fc is not None
            pwp_from_db = pwp is not None

        # Only validate against database-loaded FC/PWP, not user-supplied
        if fc is not None and pwp is not None and fc_from_db and pwp_from_db:
            if self.current_soil_moisture > fc:
                raise ValueError(f"soil_moisture must be in [0, {fc:.3f}] (fraction), got {self.current_soil_moisture}")
            if self.current_soil_moisture < pwp:
                raise ValueError(f"soil_moisture must be in [{pwp:.3f}, 1] (fraction), got {self.current_soil_moisture}")

    def _select_threshold(self) -> Optional[ThresholdReference]:
        """
        Select the appropriate threshold for operational decision.

        Priority:
        1. User-supplied MAD (explicit override)
        2. Locally validated MAD (VERIFIED_EXACT/CONTEXTUAL for this context)
        3. None - requires calibration
        """
        # 1. User override
        if self.user_mad is not None:
            return ThresholdReference(
                parameter_type="soil_water_trigger",
                parameter_name="maximum_allowable_depletion",
                parameter_value=self.user_mad,
                parameter_unit="fraction",
                evidence_level=EvidenceLevel.USER_SUPPLIED,
                verification_status="USER_SUPPLIED",
                source="User-supplied override",
                notes="User explicitly provided MAD value",
                actual_study_location="User input",
                actual_soil_context="User input",
                actual_crop_system="User input",
                is_operational=True,
            )

        # 2. Locally validated MAD
        local_mad = self._load_local_mad()
        if local_mad:
            param_value = local_mad.get("parameter_value")
            if param_value is not None:
                # Convert percent to fraction if needed
                if local_mad.get("parameter_unit") == "percent":
                    param_value = param_value / 100.0

                return ThresholdReference(
                    parameter_type="soil_water_trigger",
                    parameter_name="maximum_allowable_depletion",
                    parameter_value=param_value,
                    parameter_unit="fraction",
                    evidence_level=EvidenceLevel.RESEARCH_BACKED if local_mad["verification_status"] == "VERIFIED_EXACT" else EvidenceLevel.RESEARCH_CONTEXTUAL,
                    verification_status=local_mad["verification_status"],
                    source=local_mad["source"],
                    notes=local_mad["notes"],
                    actual_study_location=local_mad.get("actual_study_location", ""),
                    actual_soil_context=local_mad.get("actual_soil_context", ""),
                    actual_crop_system=local_mad.get("actual_crop_system", ""),
                    is_operational=True,
                )

        # 3. No local MAD - requires calibration
        self._calibration_required = True
        return None

    def _build_engine(self, mad_value: float) -> SoilWaterEngine:
        """Build SoilWaterEngine with best available parameters."""
        # FC
        fc = self.user_fc
        if fc is None:
            hydraulic = self._load_soil_hydraulic_params()
            fc = hydraulic.get("field_capacity")

        # PWP
        pwp = self.user_pwp
        if pwp is None:
            hydraulic = self._load_soil_hydraulic_params()
            pwp = hydraulic.get("permanent_wilting_point")

        # Root zone depth
        rz = self.user_root_zone_depth_cm
        if rz is None:
            rz = self._load_root_zone_depth()

        # Validate all parameters are available
        missing = []
        if fc is None:
            missing.append("field_capacity")
        if pwp is None:
            missing.append("permanent_wilting_point")
        if rz is None:
            missing.append("root_zone_depth_cm")

        if missing:
            raise ValueError(f"Missing required soil parameters: {', '.join(missing)}. These require field calibration or user-supplied values.")

        params = SoilWaterParams(
            field_capacity=fc,
            permanent_wilting_point=pwp,
            root_zone_depth_cm=rz,
            mad=mad_value,
        )
        return SoilWaterEngine(params)

    def make_decision(self) -> IrrigationDecision:
        """
        Make irrigation decision based on soil water balance.

        Returns:
            IrrigationDecision with full explanation and evidence
        """
        self._validate_inputs()

        # Load Kc value for growth stage
        self._kc_value = self._load_kc_value()

        # Load all contextual thresholds
        self._reference_thresholds = self._load_all_contextual_thresholds()

        # If crop has no reference thresholds, mark calibration required
        if not self._reference_thresholds and self.crop:
            self._calibration_required = True
            self._calibration_notes = f"No reference thresholds available for crop '{self.crop}'. Requires field calibration."

        # Select operational threshold
        self._threshold_reference = self._select_threshold()

        # Check if we have a valid operational MAD
        if self._threshold_reference is None:
            # No local MAD available
            self._calibration_required = True
            self._calibration_notes = (
                f"No locally validated MAD/p found for {self.crop} in {self.state} "
                f"({self.soil_type}). Available reference thresholds are provided for context, "
                f"but no operational trigger can be determined without field calibration."
            )

            # Still calculate soil water status for reporting
            hydraulic = self._load_soil_hydraulic_params()
            fc = self.user_fc or hydraulic.get("field_capacity")
            pwp = self.user_pwp or hydraulic.get("permanent_wilting_point")
            rz = self.user_root_zone_depth_cm or self._load_root_zone_depth()

            if fc is not None and pwp is not None and rz is not None:
                self._engine = SoilWaterParams(
                    field_capacity=fc,
                    permanent_wilting_point=pwp,
                    root_zone_depth_cm=rz,
                    mad=0.5,  # Default for calculation only
                )
                # Calculate without triggering
                ta = calculate_taw(fc, pwp, rz)
                depletion = calculate_depletion(fc, self.current_soil_moisture, rz)
                depletion_frac = calculate_depletion_fraction(fc, pwp, self.current_soil_moisture)

                return IrrigationDecision(
                    irrigation_required=False,
                    status="NO_IRRIGATION_REQUIRED",
                    current_soil_moisture=self.current_soil_moisture,
                    field_capacity=fc,
                    permanent_wilting_point=pwp,
                    root_zone_depth_cm=rz,
                    depletion_mm=depletion,
                    depletion_fraction=depletion_frac,
                    taw_mm=ta,
                    raw_mm=ta * 0.5,
                    threshold_used=None,
                    threshold_type=None,
                    threshold_parameter_type=None,
                    threshold_evidence_level=EvidenceLevel.REQUIRES_CALIBRATION,
                    threshold_verification_status="REQUIRES_FIELD_CALIBRATION",
                    threshold_source="No local MAD available",
                    threshold_notes=self._calibration_notes,
                    growth_stage=self.growth_stage,
                    kc_value=self._kc_value,
                    crop=self.crop,
                    state=self.state,
                    region=self.region,
                    soil_type=self.soil_type,
                    explanation=self._generate_explanation_no_threshold(),
                    warnings=self._warnings + ["No operational MAD threshold available - requires field calibration"],
                    calibration_required=True,
                    calibration_notes=self._calibration_notes,
                    reference_thresholds=self._reference_thresholds,
                )
            else:
                raise ValueError(
                    f"Cannot calculate soil water balance: missing soil hydraulic parameters. "
                    f"FC: {fc}, PWP: {pwp}, Root zone: {rz}"
                )

        # We have an operational threshold - build engine and calculate
        mad_value = self._threshold_reference.parameter_value
        self._engine = self._build_engine(mad_value)
        self._result = self._engine.calculate(self.current_soil_moisture)

        # Build decision
        decision = IrrigationDecision(
            irrigation_required=self._result.irrigation_triggered,
            status=self._result.status,
            current_soil_moisture=self._result.current_soil_moisture,
            field_capacity=self._result.field_capacity,
            permanent_wilting_point=self._result.permanent_wilting_point,
            root_zone_depth_cm=self._result.root_zone_depth_cm,
            depletion_mm=self._result.current_depletion_mm,
            depletion_fraction=self._result.depletion_fraction,
            taw_mm=self._result.total_available_water_mm,
            raw_mm=self._result.readily_available_water_mm,
            threshold_used=self._threshold_reference.parameter_value,
            threshold_type=self._get_threshold_type(),
            threshold_parameter_type=self._threshold_reference.parameter_type,
            threshold_evidence_level=self._threshold_reference.evidence_level,
            threshold_verification_status=self._threshold_reference.verification_status,
            threshold_source=self._threshold_reference.source,
            threshold_notes=self._threshold_reference.notes,
            threshold_actual_study_location=self._threshold_reference.actual_study_location,
            threshold_actual_soil_context=self._threshold_reference.actual_soil_context,
            threshold_actual_crop_system=self._threshold_reference.actual_crop_system,
            growth_stage=self.growth_stage,
            kc_value=self._kc_value,
            crop=self.crop,
            state=self.state,
            region=self.region,
            soil_type=self.soil_type,
            explanation=self._generate_explanation(),
            warnings=self._warnings,
            calibration_required=self._calibration_required,
            calibration_notes=self._calibration_notes,
            reference_thresholds=self._reference_thresholds,
        )

        return decision

    def _get_threshold_type(self) -> Optional[ThresholdType]:
        """Determine threshold type from reference."""
        if self._threshold_reference is None:
            return None
        name = self._threshold_reference.parameter_name
        if name == "maximum_allowable_depletion":
            return ThresholdType.MAD
        elif name == "available_soil_moisture_depletion":
            return ThresholdType.ASMD
        elif name == "iwcpe_ratio":
            return ThresholdType.IWCPE
        elif name == "soil_matric_potential":
            return ThresholdType.SOIL_MATRIC
        return None

    def _generate_explanation(self) -> str:
        """Generate human-readable explanation of the decision."""
        if self._result is None:
            return ""

        status_text = "REQUIRED" if self._result.irrigation_triggered else "NO_IRRIGATION_REQUIRED"
        depletion_pct = self._result.depletion_fraction * 100
        mad_pct = self._threshold_reference.parameter_value * 100

        lines = [
            f"Decision: IRRIGATION_{status_text}",
            "",
            f"Reason: Soil-water depletion has {'reached' if self._result.irrigation_triggered else 'not reached'} "
            f"the configured decision threshold.",
            "",
            f"Soil Water Balance:",
            f"  Field Capacity (FC) = {self._result.field_capacity:.3f} ({self._result.field_capacity*100:.1f}%)",
            f"  Permanent Wilting Point (PWP) = {self._result.permanent_wilting_point:.3f} ({self._result.permanent_wilting_point*100:.1f}%)",
            f"  Root Zone Depth = {self._result.root_zone_depth_cm:.0f} cm",
            f"  Total Available Water (TAW) = {self._result.total_available_water_mm:.1f} mm",
            f"  Readily Available Water (RAW) = {self._result.readily_available_water_mm:.1f} mm (MAD = {mad_pct:.0f}%)",
            f"  Current Soil Moisture = {self._result.current_soil_moisture:.3f} ({self._result.current_soil_moisture*100:.1f}%)",
            f"  Current Depletion = {self._result.current_depletion_mm:.1f} mm ({depletion_pct:.1f}% of TAW)",
            "",
            f"Threshold Used:",
            f"  Type: {self._threshold_reference.parameter_name}",
            f"  Value: {self._threshold_reference.parameter_value:.2f} ({self._threshold_reference.parameter_unit})",
            f"  Evidence: {self._threshold_reference.evidence_level.value}",
            f"  Verification: {self._threshold_reference.verification_status}",
            f"  Source: {self._threshold_reference.source}",
            f"  Study Location: {self._threshold_reference.actual_study_location}",
            f"  Soil Context: {self._threshold_reference.actual_soil_context}",
            f"  Crop System: {self._threshold_reference.actual_crop_system}",
        ]

        if self._kc_value:
            lines.extend([
                "",
                f"Crop Coefficient (Kc):",
                f"  Growth Stage: {self.growth_stage}",
                f"  Kc Value: {self._kc_value:.3f}",
            ])

        if self._reference_thresholds:
            lines.extend(["", "Reference Thresholds (contextual, NOT operational triggers):"])
            for ref in self._reference_thresholds:
                if ref.is_operational and ref.parameter_name == "maximum_allowable_depletion":
                    continue  # Skip the one we used
                lines.append(
                    f"  - {ref.parameter_name}: {ref.parameter_value} {ref.parameter_unit} "
                    f"({ref.evidence_level.value}) - {ref.actual_study_location} "
                    f"[{ref.actual_crop_system}]"
                )

        if self._warnings:
            lines.extend(["", "Warnings:"] + [f"  - {w}" for w in self._warnings])

        if self._calibration_required:
            lines.extend(["", f"Calibration Note: {self._calibration_notes}"])

        return "\n".join(lines)

    def _generate_explanation_no_threshold(self) -> str:
        """Generate explanation when no operational threshold is available."""
        return (
            f"Decision: NO IRRIGATION DECISION POSSIBLE\n\n"
            f"Reason: No locally validated Management Allowed Depletion (MAD/p) "
            f"is available for {self.crop} in {self.state} ({self.soil_type}).\n\n"
            f"Soil Water Status (for reference only):\n"
            f"  Current Soil Moisture = {self.current_soil_moisture:.3f} ({self.current_soil_moisture*100:.1f}%)\n"
            f"  (Full soil water balance calculation requires all soil parameters)\n\n"
            f"Available Reference Thresholds:\n" +
            "\n".join([
                f"  - {ref.parameter_name}: {ref.parameter_value} {ref.parameter_unit} "
                f"({ref.evidence_level.value}) - {ref.actual_study_location} "
                f"[{ref.actual_crop_system}]"
                for ref in self._reference_thresholds
            ]) +
            f"\n\nCalibration Required: {self._calibration_notes}"
        )


# Convenience function
def calculate_irrigation_need(
    crop: str,
    state: Optional[str] = None,
    region: Optional[str] = None,
    soil_type: Optional[str] = None,
    season: Optional[str] = None,
    growth_stage: Optional[str] = None,
    current_soil_moisture: Optional[float] = None,
    current_soil_moisture_unit: str = "fraction",
    user_mad: Optional[float] = None,
    user_fc: Optional[float] = None,
    user_pwp: Optional[float] = None,
    user_root_zone_depth_cm: Optional[float] = None,
) -> IrrigationDecision:
    """
    Convenience function to calculate irrigation need.

    Args:
        crop: Crop name
        state: Indian state
        region: Region within state
        soil_type: Soil type
        season: Growing season
        growth_stage: Crop growth stage
        current_soil_moisture: Current soil moisture
        current_soil_moisture_unit: "fraction" or "percent"
        user_mad: User-supplied MAD override
        user_fc: User-supplied FC override
        user_pwp: User-supplied PWP override
        user_root_zone_depth_cm: User-supplied root zone depth

    Returns:
        IrrigationDecision with full explanation
    """
    engine = IrrigationDecisionEngine(
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
    return engine.make_decision()


# Need to import pandas for pd.notna
import pandas as pd