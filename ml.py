"""Supervised models for RCM chloride migration coefficients (research only).

Target: non-steady-state RCM coefficient, 1e-12 m^2/s, modelled as log10.
This is NOT the 28-day natural diffusion coefficient D28.

Input CSV columns (as produced from the Excel workbook):
  reference, cement, method, fly_ash_fraction_of_binder,
  slag_fraction_of_binder, water_binder_ratio, D_RCM_1e-12_m2_per_s
  (optional: sheet, kept for traceability only)

Rows from the same source paper share a group, so the holdout and
cross-validation never place one paper's data on both sides of a split.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

RAW_TARGET = 'D_RCM_1e-12_m2_per_s'
NUM_FEATURES = ['water_binder_ratio', 'fly_ash_fraction_of_binder', 'slag_fraction_of_binder']
CAT_FEATURES = ['cement_family', 'cement_grade', 'method']
FEATURES = NUM_FEATURES + CAT_FEATURES
REQUIRED = ['reference', 'cement', 'method'] + NUM_FEATURES + [RAW_TARGET]

MODEL_NAMES = {
    'ridge': 'Ridge (tuyến tính, mốc tham chiếu)',
    'rf': 'Random Forest',
    'hgb': 'Gradient Boosting (HistGB)',
    'svr': 'SVR (RBF)',
    'gpr': 'Gaussian Process (có độ bất định)',
}
GPR_MAX_ROWS = 1500
# Chosen by repeated grouped CV on all data (ml.repeated_cv, 5x5): GPR R2 0.730 +/- 0.010,
# vs HistGB 0.697 and Ridge 0.695. Holdout on the 80/20 split ranked Ridge first; see notes.
DEFAULT_MODEL = 'gpr'
MIN_ROWS = 30
MIN_GROUPS = 5
MIN_TEST_COUNT_RELIABLE = 20
PLAUSIBLE_RANGE = (0.1, 100.0)  # 1e-12 m^2/s, rough range for RCM values

_CEMENT_RE = re.compile(r'^(PII|PI|PO)(\d{2})(?:\.?(\d))?(R?)$')


def _parse_cement(value):
    """Map spelling variants (PⅡ52R, P·O 42.5, P0 32.5, PII42∙5) to family and grade."""
    s = str(value).upper().replace('Ⅱ', 'II').replace('Ⅰ', 'I')
    s = s.replace('∙', '.').replace('·', '').replace(' ', '')
    if s.startswith('P0'):
        s = 'PO' + s[2:]
    m = _CEMENT_RE.match(s)
    if not m:
        return 'OTHER', 'OTHER'
    family, strength, decimal, early = m.groups()
    grade = f'{strength}.{decimal or "5"}{early}'
    return family, grade


def _parse_method(value):
    s = str(value).lower()
    if 'constant' in s:
        return 'constant'
    if 'variable' in s:
        return 'variable'
    return 'not_stated'


def _paper_key(reference) -> str:
    """Rough key for one source paper: letters only, first 40 characters."""
    return re.sub(r'[^a-z]', '', str(reference).lower())[:40]


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise labels, drop exact duplicates and invalid rows. Never imputes the target."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError('Dữ liệu đầu vào phải là bảng CSV')
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError('Thiếu cột CSV: ' + ', '.join(missing))

    d = df[REQUIRED].copy()
    for c in NUM_FEATURES + [RAW_TARGET]:
        d[c] = pd.to_numeric(d[c], errors='coerce')
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=NUM_FEATURES + [RAW_TARGET])

    parsed = d['cement'].map(_parse_cement)
    d['cement_family'] = [p[0] for p in parsed]
    d['cement_grade'] = [p[1] for p in parsed]
    d['method'] = d['method'].map(_parse_method)
    d['paper'] = d['reference'].map(_paper_key)

    d = d.drop_duplicates(subset=['paper', 'cement', 'method'] + NUM_FEATURES + [RAW_TARGET])
    ok = (
        d[RAW_TARGET].gt(0)
        & d['water_binder_ratio'].between(0.10, 1.50)
        & d['fly_ash_fraction_of_binder'].between(0, 1)
        & d['slag_fraction_of_binder'].between(0, 1)
        & (d['fly_ash_fraction_of_binder'] + d['slag_fraction_of_binder'] <= 1)
    )
    d = d.loc[ok].reset_index(drop=True)
    d['log10_D'] = np.log10(d[RAW_TARGET])
    return d


def _pre(scale: bool):
    return ColumnTransformer([
        ('num', StandardScaler() if scale else 'passthrough', NUM_FEATURES),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_FEATURES),
    ])


def _make_model(name: str):
    if name == 'ridge':
        return make_pipeline(_pre(True), RidgeCV(alphas=np.logspace(-3, 3, 13)))
    if name == 'rf':
        return make_pipeline(_pre(False), RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))
    if name == 'hgb':
        return make_pipeline(_pre(False), HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=300, min_samples_leaf=10,
            l2_regularization=1.0, early_stopping=False, random_state=42))
    if name == 'svr':
        return make_pipeline(_pre(True), SVR(C=10.0, epsilon=0.02, gamma='scale'))
    if name == 'gpr':
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(0.1)
        return make_pipeline(_pre(True), GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=42))
    raise ValueError(f'Mô hình không hỗ trợ: {name}')


def _cv(X: pd.DataFrame, y: np.ndarray, groups, name: str):
    """K-fold (grouped when possible) metrics on log10 of the coefficient."""
    n = len(y)
    k = min(5, n // 6)
    if k < 3:
        return None
    if groups is not None and groups.nunique() >= k:
        splits = GroupKFold(n_splits=k).split(X, y, groups.to_numpy())
    else:
        splits = KFold(n_splits=k, shuffle=True, random_state=42).split(X)
    r2s, rmses = [], []
    for tr, te in splits:
        m = _make_model(name)
        m.fit(X.iloc[tr], y[tr])
        p = m.predict(X.iloc[te])
        r2s.append(r2_score(y[te], p))
        rmses.append(np.sqrt(mean_squared_error(y[te], p)))
    return {
        'R2_log10_mean': float(np.mean(r2s)),
        'R2_log10_std': float(np.std(r2s)),
        'RMSE_log10_mean': float(np.mean(rmses)),
        'k': int(k),
    }


def _select(X, y, groups, names) -> pd.DataFrame:
    rows = []
    for name in names:
        if name == 'gpr' and len(y) > GPR_MAX_ROWS:
            continue
        cv = _cv(X, y, groups, name)
        if cv is not None:
            rows.append({'model': name, 'label': MODEL_NAMES[name], **cv})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows)
            .sort_values('R2_log10_mean', ascending=False)
            .reset_index(drop=True))


def compare_models(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped CV on all cleaned data, for exploration only (not the final holdout)."""
    clean = clean_dataset(df)
    if len(clean) < MIN_ROWS:
        raise ValueError(f'Cần ít nhất {MIN_ROWS} mẫu hợp lệ (hiện có {len(clean)})')
    return _select(clean[FEATURES], clean['log10_D'].to_numpy(), clean['paper'], MODEL_NAMES.keys())


def repeated_cv(df: pd.DataFrame, n_repeats: int = 5, k: int = 5) -> pd.DataFrame:
    """Grouped K-fold CV on ALL cleaned data, repeated with reshuffled paper-to-fold assignments.

    Use this to choose the default model (exploration). Do not use its scores
    as the final performance estimate, because the same data then drives the choice.
    Returns mean and std of R2 (log10) across repeats, most accurate first.
    """
    clean = clean_dataset(df)
    if len(clean) < MIN_ROWS:
        raise ValueError(f'Cần ít nhất {MIN_ROWS} mẫu hợp lệ (hiện có {len(clean)})')
    X = clean[FEATURES]
    y = clean['log10_D'].to_numpy()
    papers = clean['paper'].to_numpy()
    uniq = np.unique(papers)
    results = {name: [] for name in MODEL_NAMES if name != 'gpr' or len(y) <= GPR_MAX_ROWS}
    for rep in range(n_repeats):
        rng = np.random.default_rng(rep)
        relabel = dict(zip(uniq, rng.permutation(len(uniq))))
        groups = pd.Series([relabel[p] for p in papers])
        splits = list(GroupKFold(n_splits=k).split(X, y, groups.to_numpy()))
        for name in results:
            r2s = []
            for tr, te in splits:
                m = _make_model(name)
                m.fit(X.iloc[tr], y[tr])
                r2s.append(r2_score(y[te], m.predict(X.iloc[te])))
            results[name].append(float(np.mean(r2s)))
    rows = [{'model': n, 'label': MODEL_NAMES[n],
             'R2_log10_mean': float(np.mean(v)), 'R2_log10_std': float(np.std(v)),
             'repeats': n_repeats, 'folds_per_repeat': k}
            for n, v in results.items()]
    return pd.DataFrame(rows).sort_values('R2_log10_mean', ascending=False).reset_index(drop=True)


def fit_model(df: pd.DataFrame, model_name: str = DEFAULT_MODEL):
    """Return (model, metrics, bounds).

    Default is DEFAULT_MODEL (GPR). model_name='auto': pick the best model by grouped CV
    on the TRAINING split only, then score once on the 20% holdout. Any other key in
    MODEL_NAMES fits that model directly.
    """
    clean = clean_dataset(df)
    if len(clean) < MIN_ROWS:
        raise ValueError(f'Cần ít nhất {MIN_ROWS} mẫu hợp lệ (hiện có {len(clean)})')
    if model_name != 'auto' and model_name not in MODEL_NAMES:
        raise ValueError(f'model_name phải là "auto" hoặc một trong: {", ".join(MODEL_NAMES)}')

    X = clean[FEATURES]
    y = clean['log10_D'].to_numpy()
    groups = clean['paper']
    warnings = []

    if groups.nunique() >= MIN_GROUPS:
        train_ix, test_ix = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(X, y, groups))
        split = 'Theo tài liệu gốc (mỗi bài báo nằm hoàn toàn trong một phía), 80/20'
    else:
        train_ix, test_ix = train_test_split(np.arange(len(clean)), test_size=0.2, random_state=42)
        split = 'Ngẫu nhiên theo dòng, 80/20 (quá ít tài liệu để tách theo nguồn)'
        warnings.append('Số tài liệu gốc quá ít: kết quả tách ngẫu nhiên có thể bị lạc quan.')

    Xtr, Xte = X.iloc[train_ix], X.iloc[test_ix]
    ytr, yte = y[train_ix], y[test_ix]
    gtr = groups.iloc[train_ix].reset_index(drop=True)

    selection = None
    if model_name == 'auto':
        selection = _select(Xtr.reset_index(drop=True), ytr, gtr, MODEL_NAMES.keys())
        chosen = selection.iloc[0]['model'] if len(selection) else 'ridge'
        # Prefer the simpler linear model when it is within 0.02 R2 of the best (near tie).
        if len(selection) and 'ridge' in selection['model'].values:
            ridge_r2 = selection.loc[selection['model'] == 'ridge', 'R2_log10_mean'].iloc[0]
            if selection.iloc[0]['R2_log10_mean'] - ridge_r2 < 0.02:
                chosen = 'ridge'
    else:
        chosen = model_name

    model = _make_model(chosen)
    model.fit(Xtr, ytr)
    log_pred = model.predict(Xte)
    observed, predicted = 10 ** yte, 10 ** log_pred
    baseline = np.repeat(float(np.median(10 ** ytr)), len(yte))

    perm = permutation_importance(model, Xte, yte, n_repeats=20, random_state=42, scoring='r2')
    importance = pd.DataFrame({
        'feature': FEATURES,
        'importance': perm.importances_mean,
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    bounds = {
        'num': {f: (float(Xtr[f].min()), float(Xtr[f].max())) for f in NUM_FEATURES},
        'cat': {f: sorted(Xtr[f].unique().tolist()) for f in CAT_FEATURES},
    }
    median_d = float(np.median(clean[RAW_TARGET]))
    if not (PLAUSIBLE_RANGE[0] <= median_d <= PLAUSIBLE_RANGE[1]):
        warnings.append(f'Trung vị = {median_d:.2f}e-12 m²/s nằm ngoài khoảng thường gặp, kiểm tra đơn vị.')
    if len(test_ix) < MIN_TEST_COUNT_RELIABLE:
        warnings.append(
            f'Chỉ có {len(test_ix)} mẫu kiểm tra (< {MIN_TEST_COUNT_RELIABLE}): '
            'các chỉ số sai số rất nhiễu, không nên diễn giải chi tiết.')
    if selection is not None and len(selection) > 1:
        gap = selection.iloc[0]['R2_log10_mean'] - selection.iloc[1]['R2_log10_mean']
        if gap < 0.02:
            warnings.append('Hai mô hình đứng đầu gần ngang nhau trong CV: đã ưu tiên mô hình đơn giản hơn nếu phù hợp.')

    metrics = {
        'target': 'Hệ số khuếch tán clorua RCM (10⁻¹² m²/s)',
        'model': chosen,
        'model_label': MODEL_NAMES[chosen],
        'model_selection': selection,
        'n': int(len(clean)),
        'groups_total': int(groups.nunique()),
        'train_count': int(len(train_ix)),
        'test_count': int(len(test_ix)),
        'split_strategy': split,
        'MAE_1e12': float(mean_absolute_error(observed, predicted)),
        'RMSE_1e12': float(np.sqrt(mean_squared_error(observed, predicted))),
        'R2_linear': float(r2_score(observed, predicted)),
        'R2_log10': float(r2_score(yte, log_pred)),
        'baseline_MAE_1e12': float(mean_absolute_error(observed, baseline)),
        'holdout': pd.DataFrame({
            'D_RCM_thuc_te': observed,
            'D_RCM_AI': predicted,
        }).reset_index(drop=True),
        'importance': importance,
        'filtered_rows': int(len(df) - len(clean)),
        'warnings': warnings,
    }
    return model, metrics, bounds


def predict(model, values: dict, bounds=None):
    """Predict RCM coefficient (1e-12 m²/s). Returns (value, list of features outside training domain).

    values keys: water_binder_ratio, fly_ash_fraction_of_binder, slag_fraction_of_binder,
                 cement_family ('PO', 'PI', 'PII'), cement_grade ('42.5', '52.5R', ...),
                 method ('constant', 'variable', 'not_stated').
    """
    row = {f: float(values[f]) for f in NUM_FEATURES}
    row.update({f: str(values[f]) for f in CAT_FEATURES})
    df = pd.DataFrame([row], columns=FEATURES)
    outside = []
    if bounds:
        for f in NUM_FEATURES:
            lo, hi = bounds['num'][f]
            if not lo <= row[f] <= hi:
                outside.append(f)
        for f in CAT_FEATURES:
            if row[f] not in bounds['cat'][f]:
                outside.append(f)
    return float(10 ** model.predict(df)[0]), outside
