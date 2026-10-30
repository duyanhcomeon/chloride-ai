"""Exploratory, two-stage serviceability *indicator* for reinforced concrete.

Initiation is estimated separately by ``physics.initiation_years``.
Propagation is a constant-current-density, UNIFORM steel-loss scenario.

Two user-chosen end criteria are reported side by side:
  * a cover-cracking penetration depth (micrometres, assumed / to be calibrated);
  * a percentage of bar area lost (mm-scale).
Neither is failure, bond deterioration, capacity or compliance with any design
standard. Cracking normally arrives much earlier than a 10 % area loss.

0.0116 mm/year per microA/cm2 uses Faraday's law for carbon steel
(Fe -> Fe2+) and approximates ASTM G102-23's penetration conversion.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd

STEEL_RATE_MM_PER_YEAR_PER_UA_CM2 = 0.0116

# Column names shared with app.py and pdf_report.py (single source of truth).
COL_TIME = 'Thời gian từ lúc bắt đầu (năm)'
COL_LOSS = 'Hao hụt tiết diện thép (%)'
COL_REMAIN = 'Tiết diện thép còn lại (%)'
COL_ICORR = 'i_corr (µA/cm²)'
COL_T_CRACK = 'T nứt lớp bảo vệ (năm)'
COL_T_PROP = 'T phát triển ăn mòn (năm)'
COL_T_LIMIT = 'T đạt ngưỡng hao hụt (năm)'


def corrosion_rate_mm_year(current_density_uA_cm2: float) -> float:
    """Average steel surface penetration rate; assumes uniform corrosion."""
    if not math.isfinite(current_density_uA_cm2) or current_density_uA_cm2 <= 0:
        raise ValueError('i_corr must be positive and finite')
    return STEEL_RATE_MM_PER_YEAR_PER_UA_CM2 * current_density_uA_cm2


def area_loss_percent(diameter_mm: float, penetration_mm):
    """Percent of circular bar area removed by uniform radial penetration.

    ``penetration_mm`` may be scalar or NumPy array; output preserves shape.
    """
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError('bar diameter must be positive and finite')
    p = np.asarray(penetration_mm, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError('steel penetration must be finite and nonnegative')
    remain = np.maximum(diameter_mm - 2 * p, 0) / diameter_mm
    result = 100.0 * (1 - remain**2)
    return float(result) if result.ndim == 0 else result


def penetration_for_area_loss(diameter_mm: float, area_loss_limit_pct: float) -> float:
    if not math.isfinite(diameter_mm) or diameter_mm <= 0:
        raise ValueError('bar diameter must be positive and finite')
    if not math.isfinite(area_loss_limit_pct) or not (0 < area_loss_limit_pct < 100):
        raise ValueError('area loss limit must be in (0, 100) percent')
    return 0.5 * diameter_mm * (1 - math.sqrt(1 - area_loss_limit_pct / 100))


def propagation_years(
    diameter_mm: float,
    current_density_uA_cm2: float,
    area_loss_limit_pct: float,
) -> float:
    """Duration after initiation until the *chosen* steel area-loss limit."""
    required_penetration = penetration_for_area_loss(diameter_mm, area_loss_limit_pct)
    return required_penetration / corrosion_rate_mm_year(current_density_uA_cm2)


def cracking_years(current_density_uA_cm2: float, cracking_penetration_um: float) -> float:
    """Duration after initiation until an assumed cover-cracking penetration depth."""
    if not math.isfinite(cracking_penetration_um) or cracking_penetration_um <= 0:
        raise ValueError('cracking penetration must be positive and finite (um)')
    return (cracking_penetration_um / 1000.0) / corrosion_rate_mm_year(current_density_uA_cm2)


def serviceability_indicator(
    initiation_years: float | None,
    diameter_mm: float,
    current_density_uA_cm2: float,
    area_loss_limit_pct: float,
    cracking_penetration_um: float | None = None,
) -> dict:
    """Return conditional two-phase estimate; totals are None if initiation unknown."""
    phase2 = propagation_years(diameter_mm, current_density_uA_cm2, area_loss_limit_pct)
    phase1 = None if initiation_years is None else float(initiation_years)
    if phase1 is not None and (not math.isfinite(phase1) or phase1 < 0):
        raise ValueError('initiation years must be finite and nonnegative')
    crack_after = None
    crack_total = None
    crack_loss_pct = None
    if cracking_penetration_um is not None:
        crack_after = cracking_years(current_density_uA_cm2, cracking_penetration_um)
        crack_total = None if phase1 is None else phase1 + crack_after
        crack_loss_pct = area_loss_percent(diameter_mm, cracking_penetration_um / 1000.0)
    return {
        'initiation_years': phase1,
        'propagation_years': phase2,
        'total_years': None if phase1 is None else phase1 + phase2,
        'cracking_penetration_um': None if cracking_penetration_um is None else float(cracking_penetration_um),
        'cracking_years': crack_after,
        'cracking_total_years': crack_total,
        'cracking_area_loss_pct': crack_loss_pct,
        'diameter_mm': float(diameter_mm),
        'current_density_uA_cm2': float(current_density_uA_cm2),
        'area_loss_limit_pct': float(area_loss_limit_pct),
        'rate_mm_per_year': corrosion_rate_mm_year(current_density_uA_cm2),
        'penetration_limit_mm': penetration_for_area_loss(diameter_mm, area_loss_limit_pct),
    }


def remaining_bar_history(result: dict, years=None, n=220) -> pd.DataFrame:
    """Remaining area over time, starting at initiation; no corrosion before."""
    if result['total_years'] is None:
        raise ValueError('initiation time unresolved; no timeline can be shown')
    init = result['initiation_years']
    total = result['total_years']
    if years is None:
        years = np.linspace(0.0, max(total * 1.15, init + 5), n)
    times = np.asarray(years, dtype=float)
    if np.any(~np.isfinite(times)) or np.any(times < 0):
        raise ValueError('time points must be finite and nonnegative')
    penetrations = result['rate_mm_per_year'] * np.maximum(times - init, 0)
    losses = area_loss_percent(result['diameter_mm'], penetrations)
    return pd.DataFrame({
        COL_TIME: times,
        COL_LOSS: losses,
        COL_REMAIN: 100.0 - losses,
    })


def current_density_scenarios(
    initiation_years: float | None,
    diameter_mm: float,
    area_loss_limit_pct: float,
    currents=(0.25, 0.5, 1.0, 2.0, 5.0),
    cracking_penetration_um: float | None = None,
) -> pd.DataFrame:
    """Sensitivity of the end times to the assumed corrosion current density."""
    records = []
    for current in currents:
        result = serviceability_indicator(
            initiation_years, diameter_mm, current, area_loss_limit_pct, cracking_penetration_um)
        record = {COL_ICORR: current}
        if cracking_penetration_um is not None:
            record[COL_T_CRACK] = result['cracking_total_years']
        record[COL_T_PROP] = result['propagation_years']
        record[COL_T_LIMIT] = result['total_years']
        records.append(record)
    return pd.DataFrame.from_records(records)
