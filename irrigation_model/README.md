# Irrigation Module

Research-backed irrigation scheduling module for the Smart Farming Assistant.

## Purpose

This module provides a deterministic, research-backed irrigation decision engine based on Indian agricultural research. It implements:

1. **Weather input validation** - Validates meteorological sensor data
2. **Reference evapotranspiration (ET₀)** - FAO-56 Penman-Monteith (primary) and Hargreaves-Samani (fallback)
3. **Crop evapotranspiration (ETc)** - ETc = ET₀ × Kc using crop coefficients from research database
4. **Rainfall processing** - Preserves measured rainfall; effective rainfall NOT automatically calculated
5. **Soil-water balance** - MAD (Management Allowed Depletion) based irrigation triggering
6. **Irrigation decision** - Complete decision with explanation, evidence tracking, and calibration status

## Architecture

```
Sensor/Weather Input
        ↓
WeatherInput validation (weather.py)
        ↓
ET₀ calculation via calculate_et0_auto() (et0.py)
    ├─ Penman-Monteith (if all PM inputs available)
    └─ Hargreaves fallback (if Tmin/Tmax/latitude available)
        ↓
ETc calculation via calculate_etc() (etc.py)
    └─ ETc = ET₀ × Kc (Kc from database)
        ↓
Rainfall processing via process_rainfall() (rainfall.py)
    └─ rainfall_mm preserved, effective_rainfall_mm = None (NOT_CALCULATED)
        ↓
Existing irrigation decision engine (irrigation_decision.py)
    └─ Soil-water balance with MAD threshold
        ↓
Structured IrrigationAssessment result
```

## Input Requirements

### Required Inputs
| Parameter | Unit | Description |
|-----------|------|-------------|
| `crop` | str | Crop name (e.g., "soybean") |
| `current_soil_moisture` | fraction (0-1) or percent (0-100) | Current volumetric water content |
| `current_soil_moisture_unit` | "fraction" or "percent" | Unit of soil moisture |
| `growth_stage` | str | "initial", "development", "mid", "late" |

### Soil Parameters (required for decision)
| Parameter | Unit | Description |
|-----------|------|-------------|
| `user_fc` | fraction | Field capacity |
| `user_pwp` | fraction | Permanent wilting point |
| `user_root_zone_depth_cm` | cm | Root zone depth |
| `user_mad` | fraction | Management allowed depletion (optional if in DB) |

### Weather Inputs (for ET₀)
| Parameter | Unit | Penman-Monteith | Hargreaves |
|-----------|------|-----------------|------------|
| `air_temperature` | °C | ✓ | ✓ |
| `minimum_temperature` | °C | ✓ | ✓ |
| `maximum_temperature` | °C | ✓ | ✓ |
| `relative_humidity` | % | ✓ | - |
| `wind_speed` | m/s | ✓ | - |
| `solar_radiation` | MJ/m²/day | ✓ | - |
| `latitude` | degrees | ✓ | ✓ |
| `elevation` | m | ✓ | - |
| `timestamp` | datetime | ✓ | ✓ |

### Context (for Kc lookup)
| Parameter | Description |
|-----------|-------------|
| `state` | Indian state |
| `region` | Region within state |
| `soil_type` | Soil type |
| `season` | Growing season (Kharif/Rabi) |

## ET₀ Methods

### FAO-56 Penman-Monteith (Primary)
- **Inputs**: Tmean, Tmin, Tmax, RH, wind_speed, solar_radiation, latitude, elevation, day_of_year
- **Equation**: ET₀ = (0.408Δ(Rn-G) + γ(900/(T+273))u₂(es-ea)) / (Δ+γ(1+0.34u₂))
- **Status**: CALCULATED / INSUFFICIENT_DATA / INVALID_INPUT

### Hargreaves-Samani (Fallback)
- **Inputs**: Tmean, Tmin, Tmax, latitude, day_of_year
- **Equation**: ET₀ = 0.0023 × Ra × (Tmean+17.8) × √(Tmax-Tmin)
- **Auto-selected** when PM inputs insufficient but Hargreaves inputs available

## Kc/ETc Calculation

- **ETc = ET₀ × Kc**
- Kc looked up from threshold database by crop, growth_stage, state, region, soil_type, season
- **No default Kc** - returns Kc_NOT_AVAILABLE if not found
- Existing soybean Kc values (VERIFIED_CONTEXTUAL):
  - initial: 0.325
  - development: 0.750
  - mid: 1.100
  - late: 0.600

## Rainfall Handling

- Measured `rainfall_mm` preserved exactly as input
- **NO automatic effective rainfall calculation**
- `effective_rainfall_mm = None`
- `effective_rainfall_status = "NOT_CALCULATED"`
- Three-state distinction: VALID (measured), MISSING (no data), INVALID (bad data)

## Soil-Water Decision

- Uses existing `calculate_irrigation_need()` unchanged
- MAD (Management Allowed Depletion) threshold
- Decision: IRRIGATION_REQUIRED / NO_IRRIGATION_REQUIRED
- **Weather/ETc provides additional context only - does NOT create new decision rules**

## Calibration Limitations

### MP Soybean + Vertisol + No User MAD
- **Reports**: `REQUIRES_FIELD_CALIBRATION`
- **No default MAD invented** - FAO-56 p=0.50 not used as operational trigger
- Research records show MAD/p requires field calibration for this context

### Safety Rules (Preserved)
- **ASMD is NEVER converted to MAD** - remains reference parameter
- **IW:CPE is NEVER converted to ASMD** - separate scheduling parameter
- **No invented thresholds** - only research-verified values used

## Main API

```python
from irrigation.src import run_irrigation_assessment
from datetime import datetime

result = run_irrigation_assessment(
    crop="soybean",
    current_soil_moisture=0.30,
    current_soil_moisture_unit="fraction",
    growth_stage="mid",
    state="Madhya Pradesh",
    region="Bhopal/Indore region",
    soil_type="Vertisol",
    season="Kharif",
    user_mad=0.5,
    user_fc=0.382,
    user_pwp=0.246,
    user_root_zone_depth_cm=84,
    air_temperature=25.0,
    minimum_temperature=20.0,
    maximum_temperature=30.0,
    relative_humidity=65.0,
    rainfall_mm=5.0,
    wind_speed=2.5,
    solar_radiation=20.0,
    latitude=23.2,
    elevation=500.0,
    timestamp=datetime(2024, 6, 15, 12, 0),
)

final = result.to_final_dict()
```

## Example Output

```json
{
  "status": "CALIBRATION_REQUIRED",
  "irrigation_required": false,
  "crop": "soybean",
  "growth_stage": "mid",
  "soil_water": {
    "current_soil_moisture": 0.30,
    "field_capacity": 0.382,
    "permanent_wilting_point": 0.246,
    "root_zone_depth_cm": 84,
    "depletion_mm": 68.88,
    "depletion_fraction": 0.603
  },
  "weather": {
    "temperature_c": 25.0,
    "humidity_percent": 65.0,
    "rainfall_mm": 5.0,
    "wind_speed_ms": 2.5,
    "solar_radiation_mj_m2_day": 20.0
  },
  "et0": {
    "value_mm_day": 4.97,
    "method": "FAO56_PENMAN_MONTEITH",
    "status": "CALCULATED"
  },
  "crop_water_demand": {
    "kc": 1.1,
    "etc_mm_day": 5.47,
    "growth_stage": "mid"
  },
  "rainfall": {
    "rainfall_mm": 5.0,
    "effective_rainfall_mm": null,
    "effective_rainfall_status": "NOT_CALCULATED"
  },
  "calibration": {
    "required": true,
    "reason": "No locally validated MAD/p found for soybean in Madhya Pradesh (Vertisol)..."
  },
  "evidence": {
    "source": "No local MAD available",
    "verification_status": "REQUIRES_FIELD_CALIBRATION",
    "evidence_level": "REQUIRES_CALIBRATION"
  },
  "explanation": "Decision: NO IRRIGATION DECISION POSSIBLE...",
  "warnings": [
    "Effective rainfall NOT_CALCULATED - measured rainfall not automatically converted to effective rainfall"
  ]
}
```

## Test Command

```bash
python -m pytest irrigation/tests.py -vv --tb=short
```

## Current Test Count

**179 tests passing** (as of finalization)

| Category | Tests |
|----------|-------|
| SoilWaterEngine | 12 |
| ThresholdDatabase | 3 |
| ThresholdConfig | 3 |
| Preprocessing | 3 |
| DatasetInspection | 1 |
| VerificationSafety | 8 |
| ResearchContextIntegrity | 10 |
| IrrigationDecisionEngine | 21 |
| WeatherInput | 15 |
| PenmanMonteith | 14 |
| Hargreaves | 15 |
| ETc | 18 |
| Rainfall | 12 |
| PipelineIntegration | 12 |
| Finalization | 16 |
| **Total** | **179** |

## Important Notes

- **This is a deterministic irrigation baseline** - not an ML model
- **MP soybean Vertisol MAD/p requires field calibration** - no operational MAD available without user override
- **Research ASMD values are contextual references** - not converted to operational MAD
- **IW:CPE is a separate scheduling parameter** - distinct from soil-water depletion triggers
- **Effective rainfall is currently NOT calculated automatically** - measured rainfall preserved as-is
- All 19 research records preserved with full provenance tracking