"""
ET₀ Calculation Module - FAO-56 Penman-Monteith Reference Evapotranspiration.

This module implements the FAO-56 Penman-Monteith equation for reference
evapotranspiration (ET₀) calculation. All units are SI-based.

Reference: FAO Irrigation and Drainage Paper No. 56
"Crop Evapotranspiration - Guidelines for Computing Crop Water Requirements"
by Allen, R.G., Pereira, L.S., Raes, D., Smith, M. (1998)

Units used in this implementation:
- Temperature: °C
- Pressure: kPa
- Radiation: MJ/m²/day
- Wind speed: m/s (at 2m height)
- ET₀: mm/day
- Latitude: decimal degrees (positive north)
- Elevation: meters
- Time: datetime
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List
from enum import Enum
import math


class ET0Status(Enum):
    """Status of ET₀ calculation."""
    CALCULATED = "CALCULATED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    INVALID_INPUT = "INVALID_INPUT"
    CALCULATION_ERROR = "CALCULATION_ERROR"


@dataclass
class ET0Result:
    """
    Result of ET₀ calculation.
    
    Contains the calculated ET₀ value, method used, status, and metadata.
    """
    et0_mm_day: Optional[float] = None
    method: str = "FAO56_PENMAN_MONTEITH"
    status: ET0Status = ET0Status.INSUFFICIENT_DATA
    missing_inputs: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Intermediate values for debugging/verification
    ra_mm_day: Optional[float] = None           # Extraterrestrial radiation
    rs_mm_day: Optional[float] = None           # Solar radiation (MJ/m²/day)
    rso_mm_day: Optional[float] = None          # Clear-sky radiation
    rns_mm_day: Optional[float] = None          # Net shortwave radiation
    rnl_mm_day: Optional[float] = None          # Net longwave radiation
    rn_mm_day: Optional[float] = None           # Net radiation
    g_mm_day: Optional[float] = None            # Soil heat flux
    t_mean: Optional[float] = None              # Mean temperature
    es_kpa: Optional[float] = None              # Saturation vapor pressure
    ea_kpa: Optional[float] = None              # Actual vapor pressure
    delta_kpa_c: Optional[float] = None         # Slope of vapor pressure curve
    gamma_kpa_c: Optional[float] = None         # Psychrometric constant
    u2: Optional[float] = None                  # Wind speed at 2m
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "et0_mm_day": self.et0_mm_day,
            "method": self.method,
            "status": self.status.value,
            "missing_inputs": self.missing_inputs,
            "warnings": self.warnings,
            "intermediate": {
                "ra_mm_day": self.ra_mm_day,
                "rs_mm_day": self.rs_mm_day,
                "rso_mm_day": self.rso_mm_day,
                "rns_mm_day": self.rns_mm_day,
                "rnl_mm_day": self.rnl_mm_day,
                "rn_mm_day": self.rn_mm_day,
                "g_mm_day": self.g_mm_day,
                "t_mean": self.t_mean,
                "es_kpa": self.es_kpa,
                "ea_kpa": self.ea_kpa,
                "delta_kpa_c": self.delta_kpa_c,
                "gamma_kpa_c": self.gamma_kpa_c,
                "u2": self.u2,
            }
        }
    
    def is_valid(self) -> bool:
        """Check if ET₀ was successfully calculated."""
        return self.status == ET0Status.CALCULATED and self.et0_mm_day is not None


# Physical constants
STEFAN_BOLTZMANN = 4.903e-9  # MJ/(m²·day·K⁴)
LATENT_HEAT_VAPORIZATION = 2.45  # MJ/kg (at 20°C)
SPECIFIC_HEAT_AIR = 1.013e-3  # MJ/(kg·°C)
RATIO_MW_AIR_WATER = 0.622  # ratio of molecular weight of water vapor/dry air


def saturation_vapor_pressure(temperature: float) -> float:
    """
    Calculate saturation vapor pressure (es) at given temperature.
    
    FAO-56 Equation 11: es = 0.6108 * exp(17.27 * T / (T + 237.3))
    
    Args:
        temperature: Air temperature (°C)
    
    Returns:
        Saturation vapor pressure (kPa)
    """
    return 0.6108 * math.exp((17.27 * temperature) / (temperature + 237.3))


def slope_vapor_pressure_curve(temperature: float) -> float:
    """
    Calculate slope of saturation vapor pressure curve (Δ).
    
    FAO-56 Equation 13: Δ = 4098 * es / (T + 237.3)²
    
    Args:
        temperature: Air temperature (°C)
    
    Returns:
        Slope of vapor pressure curve (kPa/°C)
    """
    es = saturation_vapor_pressure(temperature)
    return (4098 * es) / ((temperature + 237.3) ** 2)


def psychrometric_constant(pressure: float) -> float:
    """
    Calculate psychrometric constant (γ).
    
    FAO-56 Equation 8: γ = 0.000665 * P
    
    Args:
        pressure: Atmospheric pressure (kPa)
    
    Returns:
        Psychrometric constant (kPa/°C)
    """
    return 0.000665 * pressure


def atmospheric_pressure(elevation: float) -> float:
    """
    Calculate atmospheric pressure (P) at given elevation.
    
    FAO-56 Equation 7: P = 101.3 * ((293 - 0.0065 * z) / 293)^5.26
    
    Args:
        elevation: Elevation above sea level (m)
    
    Returns:
        Atmospheric pressure (kPa)
    """
    return 101.3 * ((293 - 0.0065 * elevation) / 293) ** 5.26


def actual_vapor_pressure(tmin: float, tmax: float, rh_mean: float) -> float:
    """
    Calculate actual vapor pressure (ea) from Tmin, Tmax, and mean RH.
    
    FAO-56 Equation 17: ea = (es(Tmin) * RHmax/100 + es(Tmax) * RHmin/100) / 2
    When RHmean is used: ea = RHmean/100 * (es(Tmin) + es(Tmax)) / 2
    
    Args:
        tmin: Minimum temperature (°C)
        tmax: Maximum temperature (°C)
        rh_mean: Mean relative humidity (%)
    
    Returns:
        Actual vapor pressure (kPa)
    """
    es_tmin = saturation_vapor_pressure(tmin)
    es_tmax = saturation_vapor_pressure(tmax)
    return (rh_mean / 100.0) * (es_tmin + es_tmax) / 2.0


def extraterrestrial_radiation(latitude: float, day_of_year: int) -> float:
    """
    Calculate extraterrestrial radiation (Ra) for given latitude and day.
    
    FAO-56 Equation 21-25.
    
    Args:
        latitude: Latitude in decimal degrees (positive north)
        day_of_year: Day of year (1-366)
    
    Returns:
        Extraterrestrial radiation (MJ/m²/day)
    """
    # Convert latitude to radians
    lat_rad = math.radians(latitude)
    
    # Solar declination (Equation 24)
    solar_dec = 0.409 * math.sin(2 * math.pi * day_of_year / 365 - 1.39)
    
    # Sunset hour angle (Equation 25)
    sunset_hour_angle = math.acos(-math.tan(lat_rad) * math.tan(solar_dec))
    
    # Inverse relative distance Earth-Sun (Equation 23)
    dr = 1 + 0.033 * math.cos(2 * math.pi * day_of_year / 365)
    
    # Extraterrestrial radiation (Equation 21)
    # Gsc = solar constant = 0.0820 MJ/m²/min
    # Convert to MJ/m²/day: 0.0820 * 60 * 24 = 118.08
    gsc = 0.0820  # MJ/m²/min
    
    ra = (24 * 60 / math.pi) * gsc * dr * (
        sunset_hour_angle * math.sin(lat_rad) * math.sin(solar_dec) +
        math.cos(lat_rad) * math.cos(solar_dec) * math.sin(sunset_hour_angle)
    )
    
    return ra


def clear_sky_radiation(elevation: float, ra: float) -> float:
    """
    Calculate clear-sky solar radiation (Rso).
    
    FAO-56 Equation 37: Rso = (0.75 + 2e-5 * z) * Ra
    
    Args:
        elevation: Elevation (m)
        ra: Extraterrestrial radiation (MJ/m²/day)
    
    Returns:
        Clear-sky radiation (MJ/m²/day)
    """
    return (0.75 + 2e-5 * elevation) * ra


def net_shortwave_radiation(rs: float, albedo: float = 0.23) -> float:
    """
    Calculate net shortwave radiation (Rns).
    
    FAO-56 Equation 38: Rns = (1 - α) * Rs
    
    Args:
        rs: Solar radiation (MJ/m²/day)
        albedo: Surface albedo (default 0.23 for reference grass)
    
    Returns:
        Net shortwave radiation (MJ/m²/day)
    """
    return (1 - albedo) * rs


def net_longwave_radiation(
    tmin: float, tmax: float, ea: float, rs: float, rso: float
) -> float:
    """
    Calculate net longwave radiation (Rnl).
    
    FAO-56 Equation 39: Rnl = σ * (Tmax⁴ + Tmin⁴)/2 * (0.34 - 0.14*√ea) * (1.35*Rs/Rso - 0.35)
    
    Args:
        tmin: Minimum temperature (°C)
        tmax: Maximum temperature (°C)
        ea: Actual vapor pressure (kPa)
        rs: Solar radiation (MJ/m²/day)
        rso: Clear-sky radiation (MJ/m²/day)
    
    Returns:
        Net longwave radiation (MJ/m²/day)
    """
    # Convert temperatures to Kelvin
    tmax_k = tmax + 273.16
    tmin_k = tmin + 273.16
    
    # Stefan-Boltzmann term
    stefan_term = STEFAN_BOLTZMANN * (tmax_k**4 + tmin_k**4) / 2.0
    
    # Vapor pressure term
    vapor_term = 0.34 - 0.14 * math.sqrt(ea)
    
    # Radiation term (limit Rs/Rso to 1.0)
    rs_rso_ratio = min(rs / rso, 1.0) if rso > 0 else 1.0
    rad_term = 1.35 * rs_rso_ratio - 0.35
    
    return stefan_term * vapor_term * rad_term


def soil_heat_flux() -> float:
    """
    Soil heat flux (G) for daily calculations.
    
    FAO-56: G ≈ 0 for daily time steps
    
    Returns:
        Soil heat flux (MJ/m²/day)
    """
    return 0.0


def penman_monteith_et0(
    t_mean: float,
    t_min: float,
    t_max: float,
    rh_mean: float,
    wind_speed: float,
    solar_radiation: float,
    latitude: float,
    elevation: float,
    day_of_year: int,
) -> ET0Result:
    """
    Calculate reference evapotranspiration (ET₀) using FAO-56 Penman-Monteith.
    
    FAO-56 Equation 6:
    ET₀ = (0.408 * Δ * (Rn - G) + γ * (900 / (T + 273)) * u2 * (es - ea)) / (Δ + γ * (1 + 0.34 * u2))
    
    Where:
    - ET₀: reference evapotranspiration [mm/day]
    - Δ: slope of saturation vapor pressure curve [kPa/°C]
    - Rn: net radiation at crop surface [MJ/m²/day]
    - G: soil heat flux density [MJ/m²/day]
    - γ: psychrometric constant [kPa/°C]
    - T: mean daily air temperature [°C]
    - u2: wind speed at 2m height [m/s]
    - es: saturation vapor pressure [kPa]
    - ea: actual vapor pressure [kPa]
    
    Args:
        t_mean: Mean air temperature (°C)
        t_min: Minimum air temperature (°C)
        t_max: Maximum air temperature (°C)
        rh_mean: Mean relative humidity (%)
        wind_speed: Wind speed at 2m height (m/s)
        solar_radiation: Solar radiation (MJ/m²/day)
        latitude: Latitude (decimal degrees, positive north)
        elevation: Elevation (m)
        day_of_year: Day of year (1-366)
    
    Returns:
        ET0Result with ET₀ value and metadata
    """
    result = ET0Result()
    warnings = []
    
    # Validate inputs
    missing = []
    if t_mean is None: missing.append("mean_temperature")
    if t_min is None: missing.append("minimum_temperature")
    if t_max is None: missing.append("maximum_temperature")
    if rh_mean is None: missing.append("relative_humidity")
    if wind_speed is None: missing.append("wind_speed")
    if solar_radiation is None: missing.append("solar_radiation")
    if latitude is None: missing.append("latitude")
    if elevation is None: missing.append("elevation")
    if day_of_year is None: missing.append("day_of_year")
    
    if missing:
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = missing
        return result
    
    # Range validation
    if not (-50 <= t_mean <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"mean_temperature {t_mean}°C out of range"]
        return result
    if not (-50 <= t_min <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"minimum_temperature {t_min}°C out of range"]
        return result
    if not (-50 <= t_max <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"maximum_temperature {t_max}°C out of range"]
        return result
    if t_min > t_max:
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = ["minimum_temperature > maximum_temperature"]
        return result
    if not (0 <= rh_mean <= 100):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"relative_humidity {rh_mean}% out of range"]
        return result
    if wind_speed < 0:
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"wind_speed {wind_speed} m/s negative"]
        return result
    if solar_radiation < 0:
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"solar_radiation {solar_radiation} MJ/m²/day negative"]
        return result
    if not (-90 <= latitude <= 90):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"latitude {latitude}° out of range"]
        return result
    if elevation < -500:
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"elevation {elevation}m below minimum"]
        return result
    if not (1 <= day_of_year <= 366):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"day_of_year {day_of_year} out of range"]
        return result
    
    try:
        # Step 1: Atmospheric pressure (kPa)
        pressure = atmospheric_pressure(elevation)
        
        # Step 2: Psychrometric constant (γ) - Equation 8
        gamma = psychrometric_constant(pressure)
        
        # Step 3: Slope of saturation vapor pressure curve (Δ) - Equation 13
        delta = slope_vapor_pressure_curve(t_mean)
        
        # Step 4: Saturation vapor pressure (es) - mean of Tmin and Tmax
        es_tmin = saturation_vapor_pressure(t_min)
        es_tmax = saturation_vapor_pressure(t_max)
        es = (es_tmin + es_tmax) / 2.0
        
        # Step 5: Actual vapor pressure (ea) - Equation 17
        ea = actual_vapor_pressure(t_min, t_max, rh_mean)
        
        # Step 6: Extraterrestrial radiation (Ra) - Equations 21-25
        ra = extraterrestrial_radiation(latitude, day_of_year)
        
        # Step 7: Clear-sky radiation (Rso) - Equation 37
        rso = clear_sky_radiation(elevation, ra)
        
        # Step 8: Net shortwave radiation (Rns) - Equation 38
        rns = net_shortwave_radiation(solar_radiation)
        
        # Step 9: Net longwave radiation (Rnl) - Equation 39
        rnl = net_longwave_radiation(t_min, t_max, ea, solar_radiation, rso)
        
        # Step 10: Net radiation (Rn) - Equation 40
        rn = rns - rnl
        
        # Step 11: Soil heat flux (G) - negligible for daily
        g = soil_heat_flux()
        
        # Step 12: Wind speed at 2m (u2) - assume already at 2m
        # If wind speed measured at different height, use Equation 47
        u2 = wind_speed
        
        # Step 13: Penman-Monteith ET₀ - Equation 6
        numerator = 0.408 * delta * (rn - g) + gamma * (900 / (t_mean + 273)) * u2 * (es - ea)
        denominator = delta + gamma * (1 + 0.34 * u2)
        
        if denominator == 0:
            result.status = ET0Status.CALCULATION_ERROR
            result.missing_inputs = ["division by zero in denominator"]
            return result
        
        et0 = numerator / denominator
        
        # ET₀ cannot be negative (physically)
        if et0 < 0:
            warnings.append(f"Calculated ET₀ negative ({et0:.3f}), set to 0")
            et0 = 0.0
        
        # Check for unreasonably high values
        if et0 > 20:
            warnings.append(f"ET₀ unusually high: {et0:.2f} mm/day")
        
        # Populate result
        result.et0_mm_day = et0
        result.status = ET0Status.CALCULATED
        result.warnings = warnings
        
        # Store intermediate values
        result.t_mean = t_mean
        result.es_kpa = es
        result.ea_kpa = ea
        result.delta_kpa_c = delta
        result.gamma_kpa_c = gamma
        result.ra_mm_day = ra
        result.rs_mm_day = solar_radiation
        result.rso_mm_day = rso
        result.rns_mm_day = rns
        result.rnl_mm_day = rnl
        result.rn_mm_day = rn
        result.g_mm_day = g
        result.u2 = u2
        
    except Exception as e:
        result.status = ET0Status.CALCULATION_ERROR
        result.missing_inputs = [f"calculation error: {str(e)}"]
    
    return result


def calculate_et0_from_weather_input(weather_input) -> ET0Result:
    """
    Calculate ET₀ from a WeatherInput object.
    
    Extracts required fields and calls penman_monteith_et0.
    
    Args:
        weather_input: WeatherInput object with validated fields
    
    Returns:
        ET0Result
    """
    # Check if WeatherInput has required data
    if not weather_input.can_calculate_penman_monteith():
        result = ET0Result()
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = weather_input.get_missing_for_penman_monteith()
        return result
    
    if weather_input.timestamp is None:
        result = ET0Result()
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = ["timestamp"]
        return result
    
    day_of_year = weather_input.timestamp.timetuple().tm_yday
    
    return penman_monteith_et0(
        t_mean=weather_input.air_temperature,
        t_min=weather_input.minimum_temperature,
        t_max=weather_input.maximum_temperature,
        rh_mean=weather_input.relative_humidity,
        wind_speed=weather_input.wind_speed,
        solar_radiation=weather_input.solar_radiation,
        latitude=weather_input.latitude,
        elevation=weather_input.elevation,
        day_of_year=day_of_year,
    )


# =============================================================================
# HARGREAVES-SAMANI ET₀ CALCULATION
# =============================================================================

# Hargreaves constant (for Ra in MJ/m²/day converted to mm/day equivalent)
# ET₀ = 0.0023 * Ra_mm * (Tmean + 17.8) * sqrt(Tmax - Tmin)
# Ra_mm = Ra_MJ * 0.408 (conversion: 1 MJ/m² = 0.408 mm evaporation)
# So: ET₀ = 0.0023 * Ra_MJ * 0.408 * (Tmean + 17.8) * sqrt(Tmax - Tmin)
HARGREAVES_COEFFICIENT = 0.0023 * 0.408  # ≈ 0.0009384


def hargreaves_et0(
    t_mean: float,
    t_min: float,
    t_max: float,
    latitude: float,
    day_of_year: int,
) -> ET0Result:
    """
    Calculate reference evapotranspiration (ET₀) using Hargreaves-Samani equation.
    
    Hargreaves-Samani Equation (1985, FAO-56):
    ET₀ = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)
    
    Where:
    - ET₀: reference evapotranspiration [mm/day]
    - Ra: extraterrestrial radiation [mm/day] (equivalent evaporation depth)
    - Tmean: mean daily air temperature [°C]
    - Tmax: maximum daily air temperature [°C]
    - Tmin: minimum daily air temperature [°C]
    
    Note: This implementation uses Ra in MJ/m²/day from FAO-56 calculations
    and converts to mm/day using 1 MJ/m² = 0.408 mm (latent heat = 2.45 MJ/kg).
    
    Args:
        t_mean: Mean air temperature (°C)
        t_min: Minimum air temperature (°C)
        t_max: Maximum air temperature (°C)
        latitude: Latitude (decimal degrees, positive north)
        day_of_year: Day of year (1-366)
    
    Returns:
        ET0Result with ET₀ value and metadata
    """
    result = ET0Result()
    result.method = "HARGREAVES"
    warnings = []
    
    # Validate inputs
    missing = []
    if t_mean is None: missing.append("mean_temperature")
    if t_min is None: missing.append("minimum_temperature")
    if t_max is None: missing.append("maximum_temperature")
    if latitude is None: missing.append("latitude")
    if day_of_year is None: missing.append("day_of_year")
    
    if missing:
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = missing
        return result
    
    # Range validation
    if not (-50 <= t_mean <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"mean_temperature {t_mean}°C out of range"]
        return result
    if not (-50 <= t_min <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"minimum_temperature {t_min}°C out of range"]
        return result
    if not (-50 <= t_max <= 60):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"maximum_temperature {t_max}°C out of range"]
        return result
    if t_min > t_max:
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = ["minimum_temperature > maximum_temperature"]
        return result
    if not (-90 <= latitude <= 90):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"latitude {latitude}° out of range"]
        return result
    if not (1 <= day_of_year <= 366):
        result.status = ET0Status.INVALID_INPUT
        result.missing_inputs = [f"day_of_year {day_of_year} out of range"]
        return result
    
    try:
        # Calculate extraterrestrial radiation (Ra) in MJ/m²/day
        ra_mj = extraterrestrial_radiation(latitude, day_of_year)
        
        # Convert Ra to mm/day equivalent evaporation
        # 1 MJ/m² = 0.408 mm (using λ = 2.45 MJ/kg)
        ra_mm = ra_mj * 0.408
        
        # Temperature range
        temp_range = t_max - t_min
        
        # Hargreaves equation: ET₀ = 0.0023 * Ra_mm * (Tmean + 17.8) * sqrt(Tmax - Tmin)
        # Using the coefficient that includes the MJ->mm conversion
        et0 = HARGREAVES_COEFFICIENT * ra_mj * (t_mean + 17.8) * math.sqrt(temp_range)
        
        # ET₀ cannot be negative (physically)
        if et0 < 0:
            warnings.append(f"Calculated ET₀ negative ({et0:.3f}), set to 0")
            et0 = 0.0
        
        # Check for unreasonably high values
        if et0 > 20:
            warnings.append(f"ET₀ unusually high: {et0:.2f} mm/day")
        
        # Populate result
        result.et0_mm_day = et0
        result.status = ET0Status.CALCULATED
        result.warnings = warnings
        
        # Store intermediate values
        result.t_mean = t_mean
        result.ra_mm_day = ra_mj  # Store in MJ/m²/day for consistency
        # Add Hargreaves-specific intermediates
        result.rs_mm_day = None  # Not used in Hargreaves
        result.rso_mm_day = None
        result.rns_mm_day = None
        result.rnl_mm_day = None
        result.rn_mm_day = None
        result.g_mm_day = None
        result.es_kpa = None
        result.ea_kpa = None
        result.delta_kpa_c = None
        result.gamma_kpa_c = None
        result.u2 = None
        
        # Add Hargreaves-specific fields as attributes
        result.hargreaves_ra_mm = ra_mm
        result.hargreaves_temp_range = temp_range
        result.hargreaves_coefficient = HARGREAVES_COEFFICIENT
        
    except Exception as e:
        result.status = ET0Status.CALCULATION_ERROR
        result.missing_inputs = [f"calculation error: {str(e)}"]
    
    return result


def calculate_et0_from_weather_input_hargreaves(weather_input) -> ET0Result:
    """
    Calculate Hargreaves ET₀ from a WeatherInput object.
    
    Uses only the inputs required for Hargreaves:
    - t_mean, t_min, t_max, latitude, day_of_year
    
    Args:
        weather_input: WeatherInput object with validated fields
    
    Returns:
        ET0Result
    """
    # Check if WeatherInput has required data for Hargreaves
    if not weather_input.can_calculate_hargreaves():
        result = ET0Result()
        result.method = "HARGREAVES"
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = weather_input.get_missing_for_hargreaves()
        return result
    
    if weather_input.timestamp is None:
        result = ET0Result()
        result.method = "HARGREAVES"
        result.status = ET0Status.INSUFFICIENT_DATA
        result.missing_inputs = ["timestamp"]
        return result
    
    day_of_year = weather_input.timestamp.timetuple().tm_yday
    
    return hargreaves_et0(
        t_mean=weather_input.air_temperature,
        t_min=weather_input.minimum_temperature,
        t_max=weather_input.maximum_temperature,
        latitude=weather_input.latitude,
        day_of_year=day_of_year,
    )


def calculate_et0_auto(weather_input) -> ET0Result:
    """
    Automatically select and calculate ET₀ using best available method.
    
    Priority:
    1. Penman-Monteith (if all inputs available)
    2. Hargreaves (if temperature and latitude available)
    3. INSUFFICIENT_DATA
    
    Args:
        weather_input: WeatherInput object with validated fields
    
    Returns:
        ET0Result with method used and status
    """
    # Try Penman-Monteith first
    if weather_input.can_calculate_penman_monteith():
        return calculate_et0_from_weather_input(weather_input)
    
    # Try Hargreaves
    if weather_input.can_calculate_hargreaves():
        return calculate_et0_from_weather_input_hargreaves(weather_input)
    
    # Neither method can be calculated
    result = ET0Result()
    result.status = ET0Status.INSUFFICIENT_DATA
    result.missing_inputs = (
        weather_input.get_missing_for_penman_monteith() +
        weather_input.get_missing_for_hargreaves()
    )
    return result