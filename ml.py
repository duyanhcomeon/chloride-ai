"""Supervised D28 model with independent holdout metrics (research only).

IMPORTANT: d28_m2_s must be physically consistent across observations.
If replicate batches are present, provide mix_id to enforce group holdout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split

FEATURES = [
    'fly_ash_pct',
    'water_binder_ratio',
    'curing_days',
    'compressive_strength_mpa',
    'temperature_c',
]
TARGET = 'd28_m2_s'


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned records; never invent or impute missing target data."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Dữ liệu đầu vào phải là bảng CSV')
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError('Thiếu cột CSV: ' + ', '.join(missing))
    columns = FEATURES + [TARGET] + (['mix_id'] if 'mix_id' in df.columns else [])
    clean = df[columns].copy()
    for key in FEATURES + [TARGET]:
        clean[key] = pd.to_numeric(clean[key], errors='coerce')
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + [TARGET])
    ok = (
        clean[TARGET].gt(0)
        & clean['fly_ash_pct'].between(0, 100)
        & clean['water_binder_ratio'].between(0.10, 1.50)
        & clean['curing_days'].gt(0)
        & clean['compressive_strength_mpa'].gt(0)
        & clean['temperature_c'].between(-10, 90)
    )
    clean = clean.loc[ok].copy()
    return clean


# Typical chloride diffusion coefficients of concrete are ~0.1-100 x 10^-12 m2/s.
# A median far outside 1e-14..1e-10 m2/s almost always means a unit mix-up.
D28_PLAUSIBLE_RANGE = (1e-14, 1e-10)
MIN_TEST_COUNT_RELIABLE = 20


def cross_validate_log10(X: pd.DataFrame, y: np.ndarray, groups=None):
    """K-fold (group-aware when possible) R2 of log10(D28); None if not feasible.

    Gives a less noisy view than a single small 20 % holdout. Uses a lighter
    forest for speed; never used for the delivered model.
    """
    try:
        n = len(y)
        k = min(5, n // 6)
        if k < 3:
            return None
        if groups is not None and groups.nunique() >= k:
            splits = GroupKFold(n_splits=k).split(X, y, groups.to_numpy())
        else:
            splits = KFold(n_splits=k, shuffle=True, random_state=42).split(X)
        scores = []
        for tr, te in splits:
            fold = RandomForestRegressor(
                n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1)
            fold.fit(X.iloc[tr], y[tr])
            scores.append(r2_score(y[te], fold.predict(X.iloc[te])))
        return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)), 'k': int(k)}
    except Exception:
        return None


def fit_model(df: pd.DataFrame):
    """Return (model, metrics, train-domain); 20% holdout, reproducible.

    The model is fitted on log10(D28); holdout predictions remain strictly
    outside fitting. If >=5 groups via optional mix_id, split by group.
    No cross-validation or uncertainty bounds are claimed by this MVP.
    """
    clean = clean_dataset(df)
    if len(clean) < 30:
        raise ValueError(f'Cần ít nhất 30 mẫu hợp lệ (hiện có {len(clean)})')
    X, y = clean[FEATURES], np.log10(clean[TARGET].to_numpy())
    groups = None
    if 'mix_id' in clean.columns and clean['mix_id'].notna().all() and clean['mix_id'].nunique() >= 5:
        groups = clean['mix_id'].astype(str)
        train_ix, test_ix = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(X, y, groups))
        split = 'Theo nhóm cấp phối (mix_id), 80/20'
    else:
        train_ix, test_ix = train_test_split(np.arange(len(clean)), test_size=.2, random_state=42)
        split = 'Ngẫu nhiên theo dòng, 80/20 (chưa tách mẫu lặp cùng cấp phối)'

    Xtrain, Xtest = X.iloc[train_ix], X.iloc[test_ix]
    ytrain, ytest = y[train_ix], y[test_ix]
    model = RandomForestRegressor(
        n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    model.fit(Xtrain, ytrain)
    log_pred = model.predict(Xtest)
    observed, predicted = 10 ** ytest, 10 ** log_pred
    baseline = np.repeat(float(np.median(10 ** ytrain)), len(ytest))
    bounds = {f: (float(Xtrain[f].min()), float(Xtrain[f].max())) for f in FEATURES}
    warnings = []
    median_d = float(np.median(clean[TARGET]))
    if not (D28_PLAUSIBLE_RANGE[0] <= median_d <= D28_PLAUSIBLE_RANGE[1]):
        warnings.append(
            f'D28 trung vị = {median_d:.2e} m²/s nằm ngoài khoảng thường gặp '
            '(1e-14 – 1e-10 m²/s). Kiểm tra lại đơn vị: cột d28_m2_s phải tính bằng m²/s, '
            'không phải ×10⁻¹².')
    if len(test_ix) < MIN_TEST_COUNT_RELIABLE:
        warnings.append(
            f'Chỉ có {len(test_ix)} mẫu kiểm tra (< {MIN_TEST_COUNT_RELIABLE}): '
            'MAE, RMSE và R² rất nhiễu, không nên diễn giải chi tiết.')
    metrics = {
        'n': int(len(clean)),
        'train_count': int(len(train_ix)),
        'test_count': int(len(test_ix)),
        'split_strategy': split,
        'MAE_m2_s': float(mean_absolute_error(observed, predicted)),
        'RMSE_m2_s': float(np.sqrt(mean_squared_error(observed, predicted))),
        'R2_linear': float(r2_score(observed, predicted)),
        'R2_log10': float(r2_score(ytest, log_pred)),
        'baseline_MAE_m2_s': float(mean_absolute_error(observed, baseline)),
        'holdout': pd.DataFrame({
            'D28_thuc_te_m2_s': observed,
            'D28_AI_m2_s': predicted,
        }).reset_index(drop=True),
        'importance': pd.DataFrame({
            'feature': FEATURES,
            'importance': model.feature_importances_,
        }).sort_values('importance', ascending=False).reset_index(drop=True),
        'filtered_rows': int(len(df) - len(clean)),
        'groups_used': groups is not None,
        'cv_r2_log10': cross_validate_log10(X, y, groups),
        'warnings': warnings,
    }
    return model, metrics, bounds


def predict(model, values: dict, bounds=None):
    row = pd.DataFrame([{key: float(values[key]) for key in FEATURES}])
    missing_domain = [
        feat for feat in FEATURES
        if bounds and not (bounds[feat][0] <= float(values[feat]) <= bounds[feat][1])
    ]
    return float(10 ** model.predict(row)[0]), missing_domain
