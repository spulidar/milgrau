"""US Standard Atmosphere 1976 thermodynamic fallback for Level 2.

The fallback is deliberately deterministic and network-free. External atmospheric
profiles (radiosonde/ERA5) are acquired by :mod:`milgrau.io` and persisted in
Level 1; this module only provides the last-resort physical atmosphere.
"""

from __future__ import annotations

import numpy as np

_STANDARD_GRAVITY_M_S2 = 9.80665
_SPECIFIC_GAS_CONSTANT_DRY_AIR_J_KG_K = 287.05287
_EARTH_RADIUS_M = 6_356_766.0
_MAX_GEOPOTENTIAL_ALTITUDE_M = 84_852.0

# US Standard Atmosphere 1976 layer bases (geopotential altitude) and lapse
# rates. The final base is a boundary only; the seven lapse rates describe the
# layers below it.
_LAYER_BASE_GEOPOTENTIAL_M = np.array(
    [0.0, 11_000.0, 20_000.0, 32_000.0, 47_000.0, 51_000.0, 71_000.0, 84_852.0],
    dtype=np.float64,
)
_LAYER_LAPSE_RATE_K_M = np.array(
    [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020],
    dtype=np.float64,
)


def _layer_base_state() -> tuple[np.ndarray, np.ndarray]:
    """Return temperature (K) and pressure (Pa) at every USSA76 layer base."""
    temperatures = np.empty(_LAYER_BASE_GEOPOTENTIAL_M.size, dtype=np.float64)
    pressures = np.empty(_LAYER_BASE_GEOPOTENTIAL_M.size, dtype=np.float64)
    temperatures[0] = 288.15
    pressures[0] = 101_325.0

    for idx, lapse in enumerate(_LAYER_LAPSE_RATE_K_M):
        h0 = _LAYER_BASE_GEOPOTENTIAL_M[idx]
        h1 = _LAYER_BASE_GEOPOTENTIAL_M[idx + 1]
        t0 = temperatures[idx]
        p0 = pressures[idx]
        dh = h1 - h0
        if lapse == 0.0:
            t1 = t0
            p1 = p0 * np.exp(
                -_STANDARD_GRAVITY_M_S2 * dh
                / (_SPECIFIC_GAS_CONSTANT_DRY_AIR_J_KG_K * t0)
            )
        else:
            t1 = t0 + lapse * dh
            p1 = p0 * (t0 / t1) ** (
                _STANDARD_GRAVITY_M_S2
                / (_SPECIFIC_GAS_CONSTANT_DRY_AIR_J_KG_K * lapse)
            )
        temperatures[idx + 1] = t1
        pressures[idx + 1] = p1
    return temperatures, pressures


_LAYER_BASE_TEMPERATURE_K, _LAYER_BASE_PRESSURE_PA = _layer_base_state()


def geometric_to_geopotential_altitude(altitude_m: np.ndarray) -> np.ndarray:
    """Convert geometric altitude above mean sea level to geopotential altitude."""
    geometric = np.asarray(altitude_m, dtype=np.float64)
    return _EARTH_RADIUS_M * geometric / (_EARTH_RADIUS_M + geometric)


def get_standard_atmosphere(altitude_array_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return US Standard Atmosphere 1976 pressure (hPa) and temperature (K).

    ``altitude_array_m`` is geometric altitude above mean sea level. Negative
    finite altitudes are clamped to sea level. The implementation covers the
    seven USSA76 layers through 84.852 km geopotential altitude, far above the
    MILGRAU elastic-lidar processing domain. Values above that formal limit are
    rejected instead of being silently extrapolated.
    """
    altitude = np.asarray(altitude_array_m, dtype=np.float64)
    pressure_hpa = np.full(altitude.shape, np.nan, dtype=np.float64)
    temperature_k = np.full(altitude.shape, np.nan, dtype=np.float64)

    finite = np.isfinite(altitude)
    if not finite.any():
        return pressure_hpa, temperature_k

    geometric = np.maximum(altitude[finite], 0.0)
    geopotential = geometric_to_geopotential_altitude(geometric)
    if np.any(geopotential > _MAX_GEOPOTENTIAL_ALTITUDE_M):
        maximum = float(np.nanmax(geopotential))
        raise ValueError(
            "US Standard Atmosphere 1976 fallback is implemented only through "
            f"{_MAX_GEOPOTENTIAL_ALTITUDE_M:.0f} m geopotential altitude; got {maximum:.1f} m."
        )

    finite_pressure_pa = np.empty_like(geopotential)
    finite_temperature_k = np.empty_like(geopotential)
    layer_indices = np.searchsorted(
        _LAYER_BASE_GEOPOTENTIAL_M[1:], geopotential, side="right"
    )
    layer_indices = np.minimum(layer_indices, _LAYER_LAPSE_RATE_K_M.size - 1)

    for layer_idx in range(_LAYER_LAPSE_RATE_K_M.size):
        mask = layer_indices == layer_idx
        if not mask.any():
            continue
        h0 = _LAYER_BASE_GEOPOTENTIAL_M[layer_idx]
        t0 = _LAYER_BASE_TEMPERATURE_K[layer_idx]
        p0 = _LAYER_BASE_PRESSURE_PA[layer_idx]
        lapse = _LAYER_LAPSE_RATE_K_M[layer_idx]
        dh = geopotential[mask] - h0
        if lapse == 0.0:
            temperature = np.full(dh.shape, t0, dtype=np.float64)
            pressure = p0 * np.exp(
                -_STANDARD_GRAVITY_M_S2 * dh
                / (_SPECIFIC_GAS_CONSTANT_DRY_AIR_J_KG_K * t0)
            )
        else:
            temperature = t0 + lapse * dh
            pressure = p0 * (t0 / temperature) ** (
                _STANDARD_GRAVITY_M_S2
                / (_SPECIFIC_GAS_CONSTANT_DRY_AIR_J_KG_K * lapse)
            )
        finite_temperature_k[mask] = temperature
        finite_pressure_pa[mask] = pressure

    temperature_k[finite] = finite_temperature_k
    pressure_hpa[finite] = finite_pressure_pa / 100.0
    return pressure_hpa, temperature_k
