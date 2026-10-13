"""
Irrigation Need Detection Module for Smart Farming Assistant.

Research-backed irrigation scheduling based on Indian agricultural research.
Supports crop, region, soil type, season, and growth stage specific thresholds.
"""

__version__ = "0.1.0"
__author__ = "Smart Farming Assistant Team"

# Core modules
from .config import (
    BASE_DIR,
    DATA_RAW_DIR,
    DATA_PROCESSED_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    THRESHOLD_DATABASE_PATH,
    SENSOR_DATA_PATH,
    DEFAULT_SOIL_PARAMS,
    REQUIRED_SENSOR_COLUMNS,
    OPTIONAL_SENSOR_COLUMNS,
    VALID_EVIDENCE_LEVELS,
    VALID_THRESHOLD_TYPES,
    VALID_STATUS,
    UNITS,
)

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
    kpa_to_cbar,
    cbar_to_kpa,
)

from .threshold_database import ThresholdDatabase, get_database
from .threshold_config import ThresholdConfig, get_threshold_config, get_crop_profile
from .preprocessing import SensorPreprocessor, preprocess_sensor_data
from .feature_engineering import FeatureEngineer, engineer_features
from .predict import (
    IrrigationPredictor,
    ParameterSource,
    create_predictor,
)
from .irrigation_decision import (
    IrrigationDecisionEngine,
    IrrigationDecision,
    ThresholdReference,
    ThresholdType,
    EvidenceLevel,
    calculate_irrigation_need,
)

from .pipeline import (
    IrrigationAssessment,
    PipelineStatus,
    run_irrigation_assessment,
)

__all__ = [
    # Config
    "BASE_DIR",
    "DATA_RAW_DIR",
    "DATA_PROCESSED_DIR",
    "MODELS_DIR",
    "RESULTS_DIR",
    "THRESHOLD_DATABASE_PATH",
    "SENSOR_DATA_PATH",
    "DEFAULT_SOIL_PARAMS",
    "REQUIRED_SENSOR_COLUMNS",
    "OPTIONAL_SENSOR_COLUMNS",
    "VALID_EVIDENCE_LEVELS",
    "VALID_THRESHOLD_TYPES",
    "VALID_STATUS",
    "UNITS",
    # Soil water engine
    "SoilWaterEngine",
    "SoilWaterParams",
    "SoilWaterResult",
    "calculate_taw",
    "calculate_raw",
    "calculate_depletion",
    "calculate_depletion_fraction",
    "vwc_percent_to_fraction",
    "vwc_fraction_to_percent",
    "kpa_to_cbar",
    "cbar_to_kpa",
    # Threshold database
    "ThresholdDatabase",
    "get_database",
    # Threshold config
    "ThresholdConfig",
    "get_threshold_config",
    "get_crop_profile",
    # Preprocessing
    "SensorPreprocessor",
    "preprocess_sensor_data",
    # Feature engineering
    "FeatureEngineer",
    "engineer_features",
    # Predictor
    "IrrigationPredictor",
    "ParameterSource",
    "create_predictor",
    # Irrigation decision
    "IrrigationDecisionEngine",
    "IrrigationDecision",
    "ThresholdReference",
    "ThresholdType",
    "EvidenceLevel",
    "calculate_irrigation_need",
    # Pipeline (NEW - main public API)
    "IrrigationAssessment",
    "PipelineStatus",
    "run_irrigation_assessment",
]