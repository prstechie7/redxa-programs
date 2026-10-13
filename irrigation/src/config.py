"""
Configuration module for Irrigation Need Detection.
Centralizes paths and default parameters.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models" / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"

THRESHOLD_DATABASE_PATH = BASE_DIR / "threshold_database.csv"
SENSOR_DATA_PATH = DATA_RAW_DIR / "sensor_data.csv"

DEFAULT_SOIL_PARAMS = {
    "field_capacity": None,
    "permanent_wilting_point": None,
    "root_zone_depth_cm": None,
    "mad": None,
}

REQUIRED_SENSOR_COLUMNS = ["timestamp", "soil_moisture"]
OPTIONAL_SENSOR_COLUMNS = [
    "soil_temperature",
    "air_temperature",
    "air_humidity",
    "rainfall",
    "solar_radiation",
    "wind_speed",
]

VALID_EVIDENCE_LEVELS = [
    "DIRECT_RESEARCH",
    "DERIVED_FROM_RESEARCH",
    "FIELD_CALIBRATED",
    "ESTIMATED",
    "PROXY_INDIAN",
    "DIRECT_INDIAN_CONTEXTUAL",
    "PROXY_INDIAN_CONTEXTUAL",
    "REQUIRES_CALIBRATION",
]

VALID_THRESHOLD_TYPES = [
    "seasonal_crop_water_requirement",
    "available_soil_moisture_depletion",
    "soil_matric_potential",
    "irrigation_interval",
    "crop_water_requirement",
]

VALID_STATUS = [
    "IRRIGATION_REQUIRED",
    "NO_IRRIGATION_REQUIRED",
]

UNITS = {
    "volumetric_water_content": "fraction (0.30 = 30% VWC)",
    "soil_matric_potential": "kPa",
    "depth": "mm",
    "water_requirement": "mm",
    "temperature": "celsius",
    "humidity": "percent",
    "rainfall": "mm",
    "solar_radiation": "W/m2",
    "wind_speed": "m/s",
}