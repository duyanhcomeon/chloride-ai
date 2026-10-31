"""Convert 19 published accelerated lab measurements to v1.0 input schema.

For data auditing and format demonstration, NOT to claim field service years.
The output deliberately preserves study-level provenance and absent RCM.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'corrosion_initiation_19_groups.csv'
DEST = ROOT / 'research_sources' / 'ACCELERATED_19_AUDIT_ONLY.csv'

PROTOCOLS = {
    'MENENDEZ_2019': 'accelerated_electrical',
    'ZAFAR_2015': 'accelerated_high_NaCl',
    'PARK_2020': 'accelerated_salt_supply',
}


def convert(df: pd.DataFrame) -> pd.DataFrame:
    needed = ['study_id', 'mix_id', 'fly_ash_pct', 'water_binder_ratio',
              'cover_mm', 'time_to_corrosion_days', 'event_observed', 'source_url']
    if not set(needed).issubset(df.columns):
        raise ValueError('Thiếu thông số gốc để chuyển đổi nghiên cứu gia tốc.')
    if df['study_id'].isin(PROTOCOLS).sum() != len(df):
        raise ValueError('Có nghiên cứu chưa phân loại gia tốc, không đoán loại thí nghiệm.')
    out = pd.DataFrame({
        'fly_ash_pct': df['fly_ash_pct'],
        'water_binder_ratio': df['water_binder_ratio'],
        'cover_mm': df['cover_mm'],
        'exposure_class': 'laboratory_chloride',
        # Only convert units; not an acceleration factor for field service life.
        'observed_duration_years': df['time_to_corrosion_days'] / 365.25,
        'event_observed': df['event_observed'],
        'source_group': df['study_id'],
        'limit_state': 'corrosion_initiation_lab',
        'exposure_protocol': df['study_id'].map(PROTOCOLS),
        'source_reference': df['source_url'],
        'd_rcm_28_m2_s': pd.NA,
        'mix_id': df['mix_id'],
        'original_days': df['time_to_corrosion_days'],
        'source_table': df['source_table'],
        'record_level': df['record_level'],
        'data_status': 'AUDIT_ONLY_ACCELERATED_NOT_FIELD_LIFE',
    })
    return out


if __name__ == '__main__':
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(convert(pd.read_csv(SOURCE)).to_csv(DEST, index=False))
