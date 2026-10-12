"""Comparative scenarios for chloride ingress, strictly driven by provided D28.

Changing fly-ash % alone must NOT change corrosion estimates. The effect
appears only via independent D28 input per mixture or validated model output.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from physics import concentration, initiation_years

DEFAULT_SCENARIOS = pd.DataFrame({
    'Tro bay (%)': [0, 10, 20, 30, 40],
    'D28 (x10^-12 m2/s)': [12.0, 10.0, 8.5, 7.5, 7.0],
})


def run_comparison(data: pd.DataFrame, cover_mm: float, m: float, cs: float,
                   critical: float, c0: float, horizon: float) -> pd.DataFrame:
    needed = ['Tro bay (%)', 'D28 (x10^-12 m2/s)']
    if any(col not in data.columns for col in needed):
        raise ValueError('Thiếu cột tỷ lệ tro bay hoặc D28')
    if data.empty:
        raise ValueError('Cần tối thiểu một cấp phối để so sánh')
    out = []
    for _, row in data.iterrows():
        fly = pd.to_numeric(row['Tro bay (%)'], errors='coerce')
        scale = pd.to_numeric(row['D28 (x10^-12 m2/s)'], errors='coerce')
        if not (np.isfinite(fly) and 0 <= fly <= 100):
            raise ValueError('Hàm lượng tro bay phải nằm trong [0, 100] (%)')
        if not (np.isfinite(scale) and 0 < scale <= 10000):
            raise ValueError('Mỗi giá trị D28 phải > 0 và hữu hạn')
        d28 = float(scale) * 1e-12
        year = initiation_years(cover_mm, d28, m, cs, critical, c0, horizon)
        c10 = float(concentration(min(10.0, horizon), cover_mm, d28, m, cs, c0))
        out.append({
            'Tro bay (%)': float(fly),
            'D28 (x10^-12 m2/s)': float(scale),
            'Thời gian đạt ngưỡng (năm)': float(year) if year is not None else np.nan,
            'Cl- tại thép năm 10 (%)': c10,
            'Trạng thái': f'> {horizon} năm (chưa đạt ngưỡng)' if year is None else f'{year:.1f} năm',
        })
    table = pd.DataFrame(out)
    if table['Tro bay (%)'].duplicated().any():
        raise ValueError('Tỷ lệ tro bay phải khác nhau giữa các dòng')
    return table.sort_values('Tro bay (%)').reset_index(drop=True)


def ai_scenario_inputs(model, bounds, values: dict, ratios=None):
    """Only evaluate within training feature bounds; skipped ratios get a warning."""
    from ml import predict
    ratios = [0, 10, 20, 30, 40] if ratios is None else list(ratios)
    rows, skipped = [], []
    for rate in ratios:
        requested = dict(values, fly_ash_pct=float(rate))
        d, outside = predict(model, requested, bounds)
        if outside:
            skipped.append((rate, outside))
        else:
            rows.append({'Tro bay (%)': float(rate), 'D28 (x10^-12 m2/s)': d / 1e-12})
    return pd.DataFrame(rows, columns=['Tro bay (%)', 'D28 (x10^-12 m2/s)']), skipped
