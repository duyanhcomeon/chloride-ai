"""CloriLife AI v0.8: measured time-to-limit ML, no Fick assumptions.

This module is deliberately NOT trained on RCM-only data. It accepts a separate
scientifically traceable CSV where time_to_limit_years is an OBSERVED event time,
not a simulated outcome or duration of an intact structure.

Censored observations (event_observed=0) are NOT valid for standard regression;
they need a dedicated survival-analysis implementation and are rejected here.
A single limit_state is modeled at a time to avoid merging incomparable events.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from rcm_training import AUTO, MODELS, _regressor

NUMERIC_RAW = ('fly_ash_pct', 'water_binder_ratio', 'd_rcm_28_m2_s', 'cover_mm')
CATEGORICAL = ('exposure_class',)
FEATURES = ('fly_ash_pct', 'water_binder_ratio', 'rcm_x1e12', 'cover_mm', 'exposure_class')
TARGET = 'time_to_limit_years'
EVENT = 'event_observed'
GROUP = 'source_group'
STATE = 'limit_state'
REFERENCE = 'source_reference'
REQUIRED_COLUMNS = (*NUMERIC_RAW, *CATEGORICAL, TARGET, EVENT, GROUP, STATE, REFERENCE)
MIN_ROWS = 40
MIN_SOURCES = 8


class LifetimeDataError(ValueError):
    """Dataset cannot establish a scientifically defensible lifetime task."""


def schema_csv_bytes():
    """Header-only CSV; never creates fictitious lifespan observations."""
    return (','.join(REQUIRED_COLUMNS) + '\n').encode('utf-8-sig')


def available_states(raw: pd.DataFrame) -> list[str]:
    if STATE not in raw.columns:
        raise LifetimeDataError(f'Thiếu cột {STATE}. Phải định nghĩa trạng thái giới hạn cần dự báo.')
    result = raw[STATE].astype('string').str.strip().dropna()
    unique = sorted(v for v in result.unique().tolist() if v)
    if not unique:
        raise LifetimeDataError('Cột limit_state rỗng. Không rõ thời điểm nào được coi là hết tuổi thọ.')
    return unique


def clean_lifetime(raw: pd.DataFrame, limit_state: str) -> pd.DataFrame:
    """Strict validation: no auto-generated outcome labels or event-time imputation."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise LifetimeDataError('Thiếu cột dữ liệu tuổi thọ thật: ' + ', '.join(missing)
                                + '. File 215 mẫu RCM không có nhãn thời gian; không thể huấn luyện tuổi thọ.')
    df = raw.copy()
    for c in (STATE, GROUP, REFERENCE, *CATEGORICAL):
        df[c] = df[c].astype('string').str.strip()
        df[c] = df[c].replace({'': pd.NA, 'nan': pd.NA, 'None': pd.NA})
    df = df.loc[df[STATE].eq(limit_state).fillna(False)].copy()
    if df.empty:
        raise LifetimeDataError('Không có mẫu thuộc trạng thái giới hạn được chọn.')

    for c in (*NUMERIC_RAW, TARGET, EVENT):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    if df[EVENT].isna().any() or not df[EVENT].isin((0, 1)).all():
        raise LifetimeDataError('event_observed phải bằng 1 (đã xảy ra) hoặc 0 (kiểm duyệt).')
    censored = int((df[EVENT] == 0).sum())
    if censored:
        raise LifetimeDataError(
            f'Có {censored} mẫu event_observed=0 (chưa xảy ra trạng thái giới hạn). '
            'Không được coi thời gian theo dõi là tuổi thọ thật hoặc tự loại mẫu censored '
            'để huấn luyện hồi quy thông thường. Cần mô hình survival có xử lý censoring.'
        )
    numeric_valid = np.isfinite(df[list(NUMERIC_RAW) + [TARGET]].to_numpy(dtype=float)).all(axis=1)
    bounds_valid = (
        df['fly_ash_pct'].between(0, 100)
        & df['water_binder_ratio'].between(.10, 1.0)
        & df['d_rcm_28_m2_s'].gt(0)
        & df['cover_mm'].between(5, 250)
        & df[TARGET].gt(0)
    ).to_numpy(dtype=bool)
    provenance_valid = df[[GROUP, REFERENCE, *CATEGORICAL]].notna().all(axis=1).to_numpy(dtype=bool)
    valid = numeric_valid & bounds_valid & provenance_valid
    if not valid.all():
        raise LifetimeDataError(
            f'{int((~valid).sum())} dòng thiếu/không hợp lệ ở biến đầu vào, thời gian đo hoặc nguồn. '
            'Hãy sửa dữ liệu gốc; phần mềm không tự tạo nhãn hoặc thay số liệu.'
        )
    df['rcm_x1e12'] = df['d_rcm_28_m2_s'] * 1e12
    df = df.reset_index(drop=True)
    if len(df) < MIN_ROWS:
        raise LifetimeDataError(f'Cần ít nhất {MIN_ROWS} mẫu ĐÃ ĐẠT trạng thái giới hạn; hiện có {len(df)}.')
    ngroups = df[GROUP].nunique()
    if ngroups < MIN_SOURCES:
        raise LifetimeDataError(f'Cần ít nhất {MIN_SOURCES} nhóm nguồn độc lập; hiện có {ngroups}.')
    if df[TARGET].nunique() < 5:
        raise LifetimeDataError('Biến thời gian đích quá ít mức; không thể đánh giá mô hình hồi quy.')
    return df


def _pipe(name: str) -> Pipeline:
    prep = ColumnTransformer([
        ('num', StandardScaler(), list(FEATURES[:4])),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['exposure_class']),
    ], sparse_threshold=0)
    return Pipeline([('prep', prep), ('regressor', _regressor(name))])


def _metric(y_true, y_pred):
    return {
        'MAE_years': float(mean_absolute_error(y_true, y_pred)),
        'RMSE_years': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'R2': float(r2_score(y_true, y_pred)),
    }


def _backtransform(predicted_log):
    return np.maximum(0., np.expm1(predicted_log))


def train_lifetime(raw: pd.DataFrame, limit_state: str, selection: str = AUTO) -> dict:
    """Group-holdout + train-only grouped CV; supports all nine estimators."""
    if selection != AUTO and selection not in MODELS:
        raise LifetimeDataError('Thuật toán không nằm trong 9 thuật toán đã hỗ trợ.')
    data = clean_lifetime(raw, limit_state)
    x = data[list(FEATURES)].copy()
    years = data[TARGET].to_numpy(float)
    y = np.log1p(years)
    groups = data[GROUP].astype(str).to_numpy()

    train_idx, test_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(x, y, groups)
    )
    gtr, gte = groups[train_idx], groups[test_idx]
    if set(gtr) & set(gte):
        raise AssertionError('Nguồn nghiên cứu bị trùng giữa train/test.')
    xtr, xte = x.iloc[train_idx], x.iloc[test_idx]
    ytr, yte = y[train_idx], y[test_idx]
    nfolds = min(5, len(set(gtr)))
    if nfolds < 3:
        raise LifetimeDataError('Quá ít nguồn huấn luyện để kiểm định chéo có ý nghĩa.')
    cv = GroupKFold(n_splits=nfolds)
    rows = []
    for name in MODELS:
        predictions = np.full(len(train_idx), np.nan, float)
        for fold_tr, fold_va in cv.split(xtr, ytr, gtr):
            model = _pipe(name)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=ConvergenceWarning)
                model.fit(xtr.iloc[fold_tr], ytr[fold_tr])
            predictions[fold_va] = model.predict(xtr.iloc[fold_va])
        if not np.isfinite(predictions).all():
            raise LifetimeDataError('Kết quả kiểm định chéo không hợp lệ: ' + name)
        stats = _metric(np.expm1(ytr), _backtransform(predictions))
        rows.append({'Thuật toán': name,
                     'CV MAE (năm)': stats['MAE_years'],
                     'CV RMSE (năm)': stats['RMSE_years'],
                     'CV R²': stats['R2'],
                     'Số folds': nfolds})
    table = pd.DataFrame(rows).sort_values(['CV MAE (năm)', 'CV RMSE (năm)']).reset_index(drop=True)
    best = str(table.iloc[0]['Thuật toán'])
    chosen = best if selection == AUTO else selection
    model = _pipe(chosen)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        model.fit(xtr, ytr)
    pred = _backtransform(model.predict(xte))
    scores = _metric(np.expm1(yte), pred)
    holdout = pd.DataFrame({
        'source_group': gte,
        'source_reference': data.iloc[test_idx][REFERENCE].to_numpy(),
        'measured_time_to_limit_years': np.expm1(yte),
        'predicted_time_to_limit_years': pred,
    })
    model_baseline = np.full(len(test_idx), np.median(np.expm1(ytr)))
    bounds = {col: (float(xtr[col].min()), float(xtr[col].max())) for col in FEATURES[:4]}
    return {
        'model': model,
        'selected': chosen,
        'best_cv': best,
        'table': table,
        'scores': scores,
        'baseline_MAE_years': _metric(np.expm1(yte), model_baseline)['MAE_years'],
        'limit_state': limit_state,
        'holdout': holdout,
        'total_count': len(data),
        'train_count': len(train_idx),
        'test_count': len(test_idx),
        'train_sources': len(set(gtr)),
        'test_sources': len(set(gte)),
        'train_source_set': set(gtr),
        'test_source_set': set(gte),
        'bounds': bounds,
        'exposures': sorted(xtr['exposure_class'].unique().tolist()),
        'train_year_range': (float(np.min(np.expm1(ytr))), float(np.max(np.expm1(ytr)))),
        'data': data,
        'features': FEATURES,
        'target': TARGET,
    }


def predict_lifetime(result: dict, *, fly_ash_pct: float, water_binder_ratio: float,
                     d_rcm_28_m2_s: float, cover_mm: float,
                     exposure_class: str) -> float:
    """Predict time to this dataset's observed limit, with domain guards.

    A result is conditional on the source dataset and chosen limit state; the
    output is not a safety verification or unconditional structure lifespan.
    """
    row = {'fly_ash_pct': float(fly_ash_pct),
           'water_binder_ratio': float(water_binder_ratio),
           'rcm_x1e12': float(d_rcm_28_m2_s) * 1e12,
           'cover_mm': float(cover_mm),
           'exposure_class': str(exposure_class)}
    for c in FEATURES[:4]:
        low, high = result['bounds'][c]
        v = row[c]
        if not np.isfinite(v) or not low <= v <= high:
            raise LifetimeDataError(
                f'{c}: {v:g} ngoài miền quan sát của tập huấn luyện [{low:g}, {high:g}]. '
                'Không ngoại suy để công bố tuổi thọ.'
            )
    if row['exposure_class'] not in result['exposures']:
        raise LifetimeDataError('Môi trường này chưa xuất hiện trong tập huấn luyện.')
    model = result['model']
    return float(_backtransform(model.predict(pd.DataFrame([row])))[0])
