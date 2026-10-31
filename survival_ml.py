"""CloriLife-AI v1.0: censor-aware time-to-corrosion research models.

No Fick equation, artificial target labels or accelerated-to-field scaling.
Right-censored samples are used with a Cox partial likelihood or discrete-time
hazard models (RF / gradient boosting). Predictions are restricted to the
observed study protocol, limit state, exposure categories and training support.

Models are for research only, not durability design certification.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.spatial import Delaunay, QhullError
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler

VERSION = '1.0'
DURATION = 'observed_duration_years'
EVENT = 'event_observed'
GROUP = 'source_group'
STATE = 'limit_state'
PROTOCOL = 'exposure_protocol'
REFERENCE = 'source_reference'
OPTIONAL_RCM = 'd_rcm_28_m2_s'
NUMERIC = ('fly_ash_pct', 'water_binder_ratio', 'cover_mm')
CATEGORY = 'exposure_class'
REQUIRED = (*NUMERIC, CATEGORY, DURATION, EVENT, GROUP, STATE, PROTOCOL, REFERENCE)
MIN_ROWS = 40
MIN_SOURCES = 8
MIN_EVENTS = 12
MODELS = (
    'Cox PH (Ridge)',
    'Random Forest Survival',
    'Gradient Boosting Survival',
)
AUTO = 'Tự động (CV theo nguồn)'


class SurvivalDataError(ValueError):
    """Input not suitable for trustworthy time-to-event modelling."""


def schema_csv_bytes() -> bytes:
    """Empty header template: deliberately no invented observations."""
    return (','.join([*REQUIRED, OPTIONAL_RCM]) + '\n').encode('utf-8-sig')


def source_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Quick, no-model audit by study, outcome and exposure protocol."""
    required = [c for c in (STATE, PROTOCOL, GROUP, DURATION, EVENT) if c not in raw]
    if required:
        raise SurvivalDataError('Thiếu cột: ' + ', '.join(required))
    df = raw.copy()
    df[DURATION] = pd.to_numeric(df[DURATION], errors='coerce')
    df[EVENT] = pd.to_numeric(df[EVENT], errors='coerce')
    out = df.groupby([STATE, PROTOCOL], dropna=False).agg(
        samples=(DURATION, 'size'),
        events=(EVENT, lambda x: int(x.eq(1).sum())),
        censored=(EVENT, lambda x: int(x.eq(0).sum())),
        independent_sources=(GROUP, 'nunique'),
        min_followup_years=(DURATION, 'min'),
        max_followup_years=(DURATION, 'max'),
    ).reset_index()
    out['eligible_counts'] = (out['samples'].ge(MIN_ROWS) &
                              out['events'].ge(MIN_EVENTS) &
                              out['independent_sources'].ge(MIN_SOURCES))
    return out


def available_cohorts(raw: pd.DataFrame) -> list[tuple[str, str]]:
    needed = [c for c in (STATE, PROTOCOL) if c not in raw]
    if needed:
        raise SurvivalDataError('Thiếu cột ' + ', '.join(needed) +
                                ': không được trộn môi trường gia tốc và tự nhiên.')
    pairs = raw[[STATE, PROTOCOL]].dropna().astype(str).apply(lambda c: c.str.strip())
    return sorted({tuple(row) for row in pairs.to_numpy() if all(row)})


def clean_survival(raw: pd.DataFrame, state: str, protocol: str) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in raw]
    if missing:
        raise SurvivalDataError('Thiếu cột: ' + ', '.join(missing) +
                                '. Không thể lấy bảng RCM đơn thuần để tạo nhãn tuổi thọ.')
    df = raw.copy()
    for col in (GROUP, STATE, PROTOCOL, REFERENCE, CATEGORY):
        df[col] = df[col].astype('string').str.strip().replace({'': pd.NA, 'nan': pd.NA})
    df = df[df[STATE].eq(state).fillna(False) & df[PROTOCOL].eq(protocol).fillna(False)].copy()
    if df.empty:
        raise SurvivalDataError('Không có số liệu cho cặp trạng thái giới hạn / quy trình phơi nhiễm đã chọn.')
    for col in (*NUMERIC, DURATION, EVENT):
        df[col] = pd.to_numeric(df[col], errors='coerce')
    if OPTIONAL_RCM not in df:
        df[OPTIONAL_RCM] = np.nan
    else:
        df[OPTIONAL_RCM] = pd.to_numeric(df[OPTIONAL_RCM], errors='coerce')
    if not df[EVENT].isin([0, 1]).all():
        raise SurvivalDataError('event_observed phải là 0 (chưa xảy ra) hoặc 1 (đã xảy ra).')
    valid_numeric = np.isfinite(df[[*NUMERIC, DURATION]].to_numpy(dtype=float)).all(axis=1)
    valid_range = (df['fly_ash_pct'].between(0, 100) &
                   df['water_binder_ratio'].between(.10, 1.) &
                   df['cover_mm'].between(5., 250.) &
                   df[DURATION].gt(0)).fillna(False).to_numpy(dtype=bool)
    valid_source = df[[CATEGORY, GROUP, REFERENCE]].notna().all(axis=1).to_numpy(dtype=bool)
    invalid_rcm = df[OPTIONAL_RCM].notna() & (~np.isfinite(df[OPTIONAL_RCM]) |
                                                     df[OPTIONAL_RCM].le(0))
    valid = valid_numeric & valid_range & valid_source & ~invalid_rcm.to_numpy(dtype=bool)
    if not np.all(valid):
        raise SurvivalDataError(
            f'{int((~valid).sum())} dòng có thông số sai, thời gian âm/rỗng hoặc thiếu nguồn. '
            'Sửa tại dữ liệu gốc, phần mềm không tự thay nhãn.'
        )
    n = len(df)
    n_events = int(df[EVENT].sum())
    n_sources = int(df[GROUP].nunique())
    if n < MIN_ROWS or n_sources < MIN_SOURCES or n_events < MIN_EVENTS:
        raise SurvivalDataError(
            f'Chưa đủ dữ liệu để kiểm định: {n}/{MIN_ROWS} mẫu, '
            f'{n_events}/{MIN_EVENTS} sự kiện đã xảy ra, '
            f'{n_sources}/{MIN_SOURCES} nguồn độc lập. '
            'Tệp thí nghiệm gia tốc vài chục dòng không thể chứng minh tuổi thọ hiện trường.'
        )
    if df[DURATION].nunique() < 6:
        raise SurvivalDataError('Thời gian theo dõi quá ít mức để ước lượng đường cong sống sót.')
    df['rcm_x1e12'] = df[OPTIONAL_RCM] * 1e12
    return df.reset_index(drop=True)


def comparable_concordance(times, events, risk) -> tuple[float, int]:
    """Harrell's C; censored earlier observations are not comparable.

    Ties in measured times excluded; risk ties receive half credit.
    """
    t = np.asarray(times, dtype=float)
    e = np.asarray(events, dtype=int)
    r = np.asarray(risk, dtype=float)
    good = 0.
    pairs = 0
    for i in range(len(t)):
        if not e[i]:
            continue
        idx = np.flatnonzero(t > t[i])
        if not idx.size:
            continue
        v = r[i] - r[idx]
        good += float(np.count_nonzero(v > 1e-10)) + .5 * float(np.count_nonzero(abs(v) <= 1e-10))
        pairs += int(idx.size)
    return (float(good/pairs) if pairs else float('nan')), pairs


def _km(times, event_indicator):
    """Kaplan-Meier step function for event type supplied."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(event_indicator, dtype=int)
    steps = []
    values = []
    remaining = len(times)
    surv = 1.
    for tm in np.unique(np.sort(times)):
        d = int(np.sum((times == tm) & (events == 1)))
        if d:
            surv *= max(0., 1. - d/remaining)
            steps.append(float(tm))
            values.append(float(surv))
        remaining -= int(np.sum(times == tm))
    return np.asarray(steps), np.asarray(values)


def _km_at(km, when, left=False) -> float:
    t, s = km
    j = int(np.searchsorted(t, when, side='left' if left else 'right') - 1)
    return float(s[j]) if j >= 0 else 1.


def ipcw_brier(times_test, events_test, surv_predictions, horizon, times_train, events_train):
    """Single-horizon IPCW Brier; G(t) fitted ONLY on train, not holdout.

    Requires non-informative censoring and nonzero G; reports NaN if unsupported.
    """
    t = np.asarray(times_test, dtype=float)
    e = np.asarray(events_test, dtype=int)
    pred = np.clip(np.asarray(surv_predictions, dtype=float), 0., 1.)
    g = _km(times_train, 1 - np.asarray(events_train, dtype=int))
    gt = _km_at(g, horizon)
    weights = np.zeros(len(t), float)
    labels = (t > horizon).astype(float)
    for i in range(len(t)):
        if t[i] <= horizon and e[i]:
            gi = _km_at(g, t[i], left=True)
            if gi <= .05:
                return float('nan')
            weights[i] = 1. / gi
        elif t[i] > horizon:
            if gt <= .05:
                return float('nan')
            weights[i] = 1. / gt
    if not np.any(weights):
        return float('nan')
    return float(np.mean(weights * (labels-pred)**2))


def _baseline_s(horizon, times_train, events_train) -> float:
    return _km_at(_km(times_train, events_train), horizon)


@dataclass
class SurvivalPredictor:
    model_name: str
    use_rcm: bool
    prep: ColumnTransformer
    fit_model: object
    time_points: np.ndarray
    baseline: np.ndarray | None = None
    bin_edges: np.ndarray | None = None

    def _features(self, raw: pd.DataFrame) -> np.ndarray:
        return self.prep.transform(_input_frame(raw, self.use_rcm))

    def probabilities(self, raw: pd.DataFrame, horizons: np.ndarray) -> np.ndarray:
        """S(t|x); only call with horizons inside study's observed followup."""
        xx = self._features(raw)
        horizons = np.asarray(horizons, dtype=float).reshape(-1)
        if self.model_name == MODELS[0]:
            lp = np.clip(xx @ self.fit_model, -25., 25.)
            h0 = np.asarray([float(self.baseline[np.searchsorted(self.time_points, h, side='right')-1])
                             if np.searchsorted(self.time_points, h, side='right') > 0 else 0.
                             for h in horizons])
            # baseline stores cumulative Breslow hazard, NOT a KM survival curve
            return np.exp(-np.exp(lp)[:, None] * h0[None, :])
        edges = self.bin_edges
        bins = np.arange(1, len(edges), dtype=float)
        mats = []
        for k in range(len(bins)):
            xt = np.column_stack([xx, np.full(len(xx), (k + 1) / len(bins))])
            pp = self.fit_model.predict_proba(xt)[:, 1]
            mats.append(np.clip(pp, 1e-7, 1-1e-7))
        hazard = np.column_stack(mats)
        cumulative = np.cumsum(np.log1p(-hazard), axis=1)
        result = np.ones((len(xx), len(horizons)))
        for col, h in enumerate(horizons):
            if h <= 0:
                continue
            k = int(np.searchsorted(edges[1:], h, side='left'))
            k = min(k, len(edges)-2)
            previous = cumulative[:, k-1] if k else 0.
            ratio = np.clip((h - edges[k]) / max(edges[k+1]-edges[k], 1e-12), 0, 1)
            result[:, col] = np.exp(previous + np.log1p(-hazard[:, k]) * ratio)
        return np.clip(result, 0., 1.)


def _input_frame(df: pd.DataFrame, use_rcm: bool) -> pd.DataFrame:
    frame = df.copy()
    if use_rcm and 'rcm_x1e12' not in frame and OPTIONAL_RCM in frame:
        frame['rcm_x1e12'] = pd.to_numeric(frame[OPTIONAL_RCM], errors='coerce') * 1e12
    cols = [*NUMERIC, CATEGORY] + (['rcm_x1e12'] if use_rcm else [])
    if any(c not in frame for c in cols):
        raise SurvivalDataError('Thiếu đặc trưng dự báo: ' + ', '.join(c for c in cols if c not in frame))
    return frame[cols].copy()


def _fit_model(df: pd.DataFrame, model_name: str, use_rcm: bool) -> SurvivalPredictor:
    numeric = [*NUMERIC] + (['rcm_x1e12'] if use_rcm else [])
    prep = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), [CATEGORY]),
    ], sparse_threshold=0)
    xx = np.asarray(prep.fit_transform(_input_frame(df, use_rcm)), dtype=float)
    t = df[DURATION].to_numpy(float)
    e = df[EVENT].to_numpy(int)
    if model_name == MODELS[0]:
        event_times = np.sort(np.unique(t[e == 1]))
        event_masks = [(t == tm) & (e == 1) for tm in event_times]
        risk_masks = [t >= tm for tm in event_times]
        alpha = 0.8
        def loss_grad(beta):
            lp = np.clip(xx @ beta, -25., 25.)
            value = .5 * alpha * float(beta @ beta)
            grad = alpha * beta.copy()
            for em, rm in zip(event_masks, risk_masks):
                n_event = int(em.sum())
                risk = lp[rm]
                v = logsumexp(risk)
                weights = np.exp(risk-v)
                value += n_event * float(v) - float(lp[em].sum())
                grad += n_event * (weights @ xx[rm]) - xx[em].sum(axis=0)
            return value, grad
        sol = minimize(loss_grad, np.zeros(xx.shape[1]), jac=True, method='L-BFGS-B',
                       options={'maxiter': 400, 'ftol': 1e-9})
        if not sol.success and np.linalg.norm(sol.jac) > 1e-3:
            raise SurvivalDataError('Cox không hội tụ; dữ liệu có thể quá ít hoặc mất cân bằng.')
        beta = sol.x
        lp = np.clip(xx @ beta, -25., 25.)
        hazards = []
        for em, rm in zip(event_masks, risk_masks):
            risk_denom = np.exp(logsumexp(lp[rm]))
            hazards.append(float(em.sum()) / risk_denom)
        cumulative = np.cumsum(hazards)
        return SurvivalPredictor(model_name, use_rcm, prep, beta, event_times, cumulative)

    # Person-period expansion: censored inside a bin is OMITTED for that bin.
    unique_events = t[e == 1]
    quantiles = np.quantile(unique_events, np.linspace(0, 1, 10))
    edges = np.unique(np.r_[0., quantiles, np.max(t)])
    edges = edges[edges >= 0.]
    if len(edges) < 4:
        raise SurvivalDataError('Quá ít mốc thời gian quan sát được để học nguy cơ theo kỳ.')
    feat_rows = []
    responses = []
    for i, duration in enumerate(t):
        for k in range(1, len(edges)):
            left, right = edges[k-1], edges[k]
            if e[i] == 1 and left < duration <= right:
                feat_rows.append(np.r_[xx[i], k/(len(edges)-1)])
                responses.append(1)
                break
            elif duration >= right:
                feat_rows.append(np.r_[xx[i], k/(len(edges)-1)])
                responses.append(0)
            else:
                break
    yy = np.asarray(responses, dtype=int)
    if len(np.unique(yy)) != 2:
        raise SurvivalDataError('Dữ liệu theo kỳ không có đủ cả sự kiện và kỳ chưa sự kiện.')
    rows = np.asarray(feat_rows)
    if model_name == MODELS[1]:
        estimator = RandomForestClassifier(n_estimators=110, max_depth=8,
                                           min_samples_leaf=6, n_jobs=-1,
                                           random_state=42)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7,
                                                    min_samples_leaf=12,
                                                    learning_rate=.05, l2_regularization=1.,
                                                    random_state=42)
    estimator.fit(rows, yy)
    return SurvivalPredictor(model_name, use_rcm, prep, estimator,
                             edges[1:].copy(), bin_edges=edges)


def _evaluation(model, validation, train, horizon):
    followup = validation[DURATION].to_numpy(float)
    events = validation[EVENT].to_numpy(int)
    p = model.probabilities(validation, np.array([horizon]))[:, 0]
    cidx, pairs = comparable_concordance(followup, events, 1-p)
    brier = ipcw_brier(followup, events, p, horizon,
                       train[DURATION].to_numpy(float), train[EVENT].to_numpy(int))
    base = _baseline_s(horizon, train[DURATION], train[EVENT])
    bb = ipcw_brier(followup, events, np.full(len(p), base), horizon,
                    train[DURATION].to_numpy(float), train[EVENT].to_numpy(int))
    return {'C_index': cidx, 'comparable_pairs': pairs,
            'IPCW_Brier': brier, 'baseline_IPCW_Brier': bb,
            'survival_probability': p}


def _split_by_sources(df: pd.DataFrame):
    splitter = GroupShuffleSplit(n_splits=100, test_size=.25, random_state=2026)
    for itr, ite in splitter.split(df, df[EVENT], df[GROUP]):
        tr, te = df.iloc[itr], df.iloc[ite]
        if (len(tr) >= 28 and tr[EVENT].sum() >= 9 and te[EVENT].sum() >= 3 and
                tr[GROUP].nunique() >= 6 and te[GROUP].nunique() >= 2):
            return tr.copy().reset_index(drop=True), te.copy().reset_index(drop=True)
    raise SurvivalDataError('Không chia được nguồn train/holdout với đủ số sự kiện. '
                            'Cần bổ sung các nghiên cứu độc lập có ghi nhận ăn mòn.')


def train_survival(raw: pd.DataFrame, limit_state: str, exposure_protocol: str,
                   selection: str = AUTO) -> dict:
    """Group holdout + train-only GroupKFold; censored never treated as events."""
    if selection != AUTO and selection not in MODELS:
        raise SurvivalDataError('Thuật toán không hợp lệ.')
    df = clean_survival(raw, limit_state, exposure_protocol)
    tr, te = _split_by_sources(df)
    use_rcm = bool(df[OPTIONAL_RCM].notna().all())
    # Do not silently impute or synthesize an RCM measurement.
    exposure_seen = sorted(tr[CATEGORY].astype(str).unique().tolist())
    unseen_exposures = sorted(set(te[CATEGORY].astype(str)) - set(exposure_seen))
    horizon = float(np.quantile(tr.loc[tr[EVENT].eq(1), DURATION], .65))
    cv = GroupKFold(n_splits=3)
    folds = list(cv.split(tr, tr[EVENT], tr[GROUP]))
    scores = []
    for name in MODELS:
        c_values, b_values = [], []
        for i, j in folds:
            tdf, vdf = tr.iloc[i], tr.iloc[j]
            if tdf[EVENT].sum() < 5 or vdf[EVENT].sum() < 2:
                continue
            try:
                fit = _fit_model(tdf, name, use_rcm)
                h = float(np.quantile(tdf.loc[tdf[EVENT].eq(1), DURATION], .65))
                ev = _evaluation(fit, vdf, tdf, h)
            except (ValueError, FloatingPointError, SurvivalDataError):
                continue
            if np.isfinite(ev['C_index']) and ev['comparable_pairs'] >= 4:
                c_values.append(ev['C_index'])
            if np.isfinite(ev['IPCW_Brier']):
                b_values.append(ev['IPCW_Brier'])
        scores.append({'Mô hình': name,
                       'CV C-index': float(np.mean(c_values)) if len(c_values) >= 2 else np.nan,
                       'CV IPCW Brier': float(np.mean(b_values)) if len(b_values) >= 2 else np.nan,
                       'CV folds hợp lệ': len(c_values)})
    table = pd.DataFrame(scores)
    usable = table[np.isfinite(table['CV C-index'])]
    if usable.empty:
        raise SurvivalDataError('Chưa thể kiểm định chéo survival theo nguồn: thiếu nguồn có sự kiện.')
    best = usable.sort_values(['CV C-index', 'CV IPCW Brier'], ascending=[False, True]).iloc[0]['Mô hình']
    chosen = best if selection == AUTO else selection
    if not np.isfinite(table.loc[table['Mô hình'].eq(chosen), 'CV C-index'].iloc[0]):
        raise SurvivalDataError('Mô hình chọn không có đủ folds kiểm định độc lập.')
    model = _fit_model(tr, chosen, use_rcm)
    ev = _evaluation(model, te, tr, horizon)
    hold = te[[GROUP, REFERENCE, DURATION, EVENT, CATEGORY]].copy()
    hold['S_at_test_horizon'] = ev['survival_probability']
    hold['risk_at_test_horizon'] = 1 - ev['survival_probability']
    hold['test_horizon_years'] = horizon
    bounds = {key: (float(tr[key].min()), float(tr[key].max())) for key in NUMERIC}
    if use_rcm:
        bounds['rcm_x1e12'] = (float(tr['rcm_x1e12'].min()), float(tr['rcm_x1e12'].max()))
    good = (np.isfinite(ev['C_index']) and ev['C_index'] >= .55 and
            ev['comparable_pairs'] >= 20 and np.isfinite(ev['IPCW_Brier']) and
            np.isfinite(ev['baseline_IPCW_Brier']) and
            ev['IPCW_Brier'] < ev['baseline_IPCW_Brier'] and not unseen_exposures)
    return dict(model=model, selected=chosen, cv_best=best,
                table=table.sort_values('CV C-index', ascending=False),
                holdout=hold, evaluation=ev, horizon=horizon, pass_quality=bool(good),
                model_uses_rcm=use_rcm, max_years=float(tr[DURATION].max()),
                train_count=len(tr), test_count=len(te),
                train_events=int(tr[EVENT].sum()), test_events=int(te[EVENT].sum()),
                train_sources=tr[GROUP].nunique(), test_sources=te[GROUP].nunique(),
                train_source_set=set(tr[GROUP]), test_source_set=set(te[GROUP]),
                source_data_count=len(df), censored_count=int(df[EVENT].eq(0).sum()),
                limit_state=limit_state, exposure_protocol=exposure_protocol,
                bounds=bounds, exposures=exposure_seen,
                train_domain=tr[['fly_ash_pct', 'water_binder_ratio', CATEGORY]].copy(),
                unseen_exposures_on_holdout=unseen_exposures,
                reference_horizon=horizon)


def _guard_inputs(result, frame):
    if result['model_uses_rcm'] and 'rcm_x1e12' not in frame and OPTIONAL_RCM in frame:
        frame['rcm_x1e12'] = pd.to_numeric(frame[OPTIONAL_RCM], errors='coerce') * 1e12
    for col, (lo, hi) in result['bounds'].items():
        if col not in frame:
            raise SurvivalDataError(f'Chưa có {col} cho mô hình đã huấn luyện.')
        a = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
        if not np.isfinite(a).all() or np.any(a < lo - 1e-9) or np.any(a > hi + 1e-9):
            raise SurvivalDataError(f'{col} ngoài miền mẫu huấn luyện [{lo:g}, {hi:g}].')
    if not set(frame[CATEGORY].astype(str)).issubset(result['exposures']):
        raise SurvivalDataError('Môi trường chưa từng xuất hiện trong dữ liệu huấn luyện.')


def predict_survival(result: dict, frame: pd.DataFrame,
                     horizons: list[float] | np.ndarray) -> pd.DataFrame:
    if not result.get('pass_quality'):
        raise SurvivalDataError('Tập kiểm tra chưa vượt điều kiện chất lượng; khóa dự báo số năm.')
    hz = np.asarray(horizons, float)
    if hz.ndim != 1 or len(hz) < 1 or not np.isfinite(hz).all():
        raise SurvivalDataError('Mốc năm không hợp lệ.')
    if np.any(hz <= 0) or np.any(hz > result['max_years'] + 1e-9):
        raise SurvivalDataError(f'Chỉ hỗ trợ thời gian 0 < t <= {result["max_years"]:.2f} '
                                'năm theo dõi thực tế trong tập huấn luyện.')
    frame = frame.copy()
    _guard_inputs(result, frame)
    p = result['model'].probabilities(frame, hz)
    # Long format for downloadable charts and scenario comparisons.
    return pd.DataFrame([
        {'case': int(i+1), 'year': float(h), 'survival_probability': float(p[i, j]),
         'event_probability': float(1-p[i, j])}
        for i in range(len(frame)) for j, h in enumerate(hz)
    ])


def median_time(result, frame: pd.DataFrame) -> list[float | None]:
    """Restricted time by which predicted survival reaches 50%; None if unsupported."""
    if not result.get('pass_quality'):
        raise SurvivalDataError('Tập kiểm tra chưa đạt.')
    frame = frame.copy()
    _guard_inputs(result, frame)
    time_points = np.unique(np.r_[np.linspace(0., result['max_years'], 151),
                                   result['model'].time_points])
    surv = result['model'].probabilities(frame, time_points)
    result_median = []
    for row in surv:
        idx = np.flatnonzero(row <= .5)
        result_median.append(float(time_points[idx[0]]) if idx.size else None)
    return result_median


def inverse_survival(result: dict, *, target_years: float, minimum_survival: float,
                     cover_mm: float, exposure_class: str, rcm_predictor=None,
                     max_candidates: int = 12) -> pd.DataFrame:
    """Research search: maximize P(no observed event by target), in-domain only.

    Requires a supplied RCM predictor if the survival model was trained with RCM;
    RCM must never be held constant while sweeping the mixture unless justified.
    """
    if not result.get('pass_quality'):
        raise SurvivalDataError('Chưa được phép chạy bài toán ngược khi holdout không đạt.')
    if not 0 < minimum_survival <= 1:
        raise SurvivalDataError('Xác suất mục tiêu cần thuộc (0,1].')
    if result['model_uses_rcm'] and rcm_predictor is None:
        raise SurvivalDataError('Cần mô hình RCM dự báo cho từng cấp phối trước khi tìm ngược.')
    lf, hf = result['bounds']['fly_ash_pct']
    lw, hw = result['bounds']['water_binder_ratio']
    rows = []
    for fa in np.linspace(lf, hf, 13):
        for wb in np.linspace(lw, hw, 9):
            r = dict(fly_ash_pct=float(fa), water_binder_ratio=float(wb),
                     cover_mm=float(cover_mm), exposure_class=str(exposure_class))
            if result['model_uses_rcm']:
                try:
                    rcm = float(rcm_predictor(fa, wb))
                    r['rcm_x1e12'] = rcm * 1e12
                except (ValueError, SurvivalDataError):
                    continue
            rows.append(r)
    if not rows:
        raise SurvivalDataError('Không có ứng viên cấp phối hợp lệ, kiểm tra miền RCM.')
    frame = pd.DataFrame(rows)
    acceptable = np.ones(len(frame), bool)
    for col, (lo, hi) in result['bounds'].items():
        acceptable &= frame[col].between(lo, hi).fillna(False).to_numpy(dtype=bool)
    frame = frame.loc[acceptable].reset_index(drop=True)
    if frame.empty:
        raise SurvivalDataError('Không có phương án nào nằm trong miền đã huấn luyện.')
    # The inverse problem only searches measured 2D mixture support (same
    # environment), not arbitrary rectangles formed by marginal min/max.
    domain = result['train_domain']
    domain = domain[domain[CATEGORY].eq(exposure_class)]
    measured = np.unique(domain[['fly_ash_pct', 'water_binder_ratio']].to_numpy(float), axis=0)
    if len(measured) < 3:
        raise SurvivalDataError('Môi trường này có dưới 3 cấp phối huấn luyện khác nhau. '
                                'Không thể xác định bao lồi để tìm nghiệm ngược.')
    low = measured.min(axis=0)
    scale = np.maximum(measured.max(axis=0)-low, 1e-8)
    measured_norm = (measured-low)/scale
    candidate = frame[['fly_ash_pct', 'water_binder_ratio']].to_numpy(float)
    candidate_norm = (candidate-low)/scale
    try:
        triangulation = Delaunay(measured_norm)
        in_hull = triangulation.find_simplex(candidate_norm, tol=1e-8) >= 0
    except QhullError as exc:
        raise SurvivalDataError('Dữ liệu cấp phối cùng môi trường gần như thẳng hàng; '
                                'không thể đề xuất nghiệm hai biến.') from exc
    nearest = np.min(np.linalg.norm(candidate_norm[:, None, :]-measured_norm[None, :, :], axis=2), axis=1)
    frame = frame.loc[in_hull & (nearest <= .8)].reset_index(drop=True)
    if frame.empty:
        raise SurvivalDataError('Không tìm được ứng viên trong bao lồi và khoảng cách '
                                'hỗ trợ của các cấp phối thực nghiệm cùng môi trường.')
    probability = predict_survival(result, frame, [target_years])['survival_probability'].to_numpy()
    frame['P_khong_an_mon_truoc_muc_tieu'] = probability
    frame['dat_nguong_xac_suat'] = probability >= minimum_survival
    frame['moc_nam_nghien_cuu'] = float(target_years)
    frame['trang_thai'] = 'THAM_KHAO_NGHIEN_CUU_KHONG_PHAI_THIET_KE'
    return frame.sort_values(['dat_nguong_xac_suat', 'P_khong_an_mon_truoc_muc_tieu'],
                             ascending=[False, False]).head(max_candidates).reset_index(drop=True)
