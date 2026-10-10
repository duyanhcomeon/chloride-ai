"""Screening model for 1-D chloride ingress. Not a structural life assessment."""
import math
import numpy as np
from scipy.special import erfc
SECONDS_PER_YEAR = 365.25 * 24 * 3600
REFERENCE_YEARS = 28.0 / 365.25

def cumulative_diffusivity(t_years, d28, m=0.35):
    """Integral_0^t D(s) ds, D(s)=D28 for s<=28d else D28*(28d/s)^m.
    D28 in m2/s; result in m2. m in [0,1) for this MVP.
    """
    t = np.asarray(t_years, dtype=float)
    if d28 <= 0 or not (0 <= m < 1) or np.any(t < 0):
        raise ValueError('D28 > 0, 0 <= m < 1, t >= 0 required')
    early = np.minimum(t, REFERENCE_YEARS)
    late = np.maximum(t, REFERENCE_YEARS)
    base = early
    after = (REFERENCE_YEARS**m) * ((late**(1-m) - REFERENCE_YEARS**(1-m)) / (1-m))
    return d28 * SECONDS_PER_YEAR * (base + after)

def concentration(t_years, cover_mm, d28, m, cs, c0=0.0):
    """Cl- mass fraction, consistent basis for cs,c0, output; idealized semi-infinite case."""
    if cover_mm <= 0 or not (0 <= c0 < cs):
        raise ValueError('cover > 0 and 0 <= C0 < Cs required')
    acc = cumulative_diffusivity(t_years, d28, m)
    depth = cover_mm / 1000
    return c0 + (cs-c0)*erfc(depth / (2*np.sqrt(np.maximum(acc, 1e-300))))

def initiation_years(cover_mm, d28, m, cs, critical, c0=0, horizon=150):
    """First crossing of chloride threshold, or None if beyond horizon."""
    if not (c0 < critical < cs):
        raise ValueError('Require C0 < Ccrit < Cs')
    if horizon <= 0:
        raise ValueError('horizon must be >0')
    if concentration(horizon, cover_mm, d28, m, cs, c0) < critical:
        return None
    lo, hi = 0.0, float(horizon)
    for _ in range(80):
        mid = (lo+hi)/2
        if concentration(mid, cover_mm, d28, m, cs, c0) >= critical:
            hi = mid
        else:
            lo = mid
    return hi
