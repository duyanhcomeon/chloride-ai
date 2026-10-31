"""Bounded inverse screening for 28-day chloride migration (RCM) models.

Inverse problem: given D_RCM,28 <= D_target, propose multiple candidate combinations
of fly ash percentage and water/binder ratio for a FIXED observed cement/FA grade.

This is a source-aware, constrained GRID SEARCH of an already-trained forward
regression model; it is not an experimental mix-design method or a guarantee of
achieving target resistance.  It uses TRAINING rows only; holdout data are not
consulted when generating/designing candidate mixes.

New points are accepted only inside the 2D convex hull of real training mixes
sharing the same categorical materials, and near at least one observation.
No Fick diffusion, corrosion or structural service-life model is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay, QhullError

from rcm_training import FEATURES, NUM_COLS, TARGET

PREFERENCES = (
    'Gần ngưỡng RCM mục tiêu',
    'Ít tro bay hơn',
    'N/CKD thấp hơn',
    'Ưu tiên cấp phối đã thí nghiệm',
)
MODES = (
    'Tìm đồng thời tro bay và N/CKD',
    'Tìm tro bay, cố định N/CKD',
    'Tìm N/CKD, cố định tro bay',
)


def training_pairs(result, min_records: int = 3) -> pd.DataFrame:
    """Actual (cement, FA grade) pairs in TRAIN ONLY, never independently paired.

    For a sparse group with fewer than min_records, inverse optimization is
    unavailable; such a group is too weakly supported to present as a design.
    """
    frame = result['training_rows']
    pairs = (frame.groupby(['cement_type', 'fly_ash_grade'])
             .size().reset_index(name='n_training_rows')
             .sort_values('n_training_rows', ascending=False).reset_index(drop=True))
    return pairs.loc[pairs['n_training_rows'].ge(min_records)].copy()


def inverse_search(result, target_rcm_x1e12: float, cement_type: str,
                   fly_ash_grade: str, *, fly_ash_range=None, wb_range=None,
                   mode: str = MODES[0], fixed_fly_ash=None, fixed_wb=None,
                   preference: str = PREFERENCES[0], include_interpolations=True,
                   buffer_mae_multiple: float = 0.0, top_n: int = 8,
                   grid_steps: int = 37) -> dict:
    """Find traceable candidates satisfying a predicted RCM upper limit.

    Result keys: status, shortlisted, all_feasible, candidate_count,
    feasible_count, screening_limit_x1e12, target_rcm_x1e12, etc.
    status is 'feasible' or 'no_feasible'; the latter includes *non-passing*
    near misses for inspection but never claims compliance.

    The training corpus and model come from rcm_training.train_compare.
    """
    if mode not in MODES:
        raise ValueError('Chế độ bài toán ngược chưa hỗ trợ.')
    if preference not in PREFERENCES:
        raise ValueError('Tiêu chí xếp hạng không hợp lệ.')
    target = float(target_rcm_x1e12)
    if not np.isfinite(target) or target <= 0:
        raise ValueError('RCM mục tiêu phải là số dương có đơn vị ×10⁻¹² m²/s.')
    if buffer_mae_multiple not in (0.0, 0.5, 1.0):
        raise ValueError('Biên sàng lọc chỉ có thể là 0, 0,5 hoặc 1 lần MAE holdout.')
    if not (1 <= int(top_n) <= 30):
        raise ValueError('Số phương án phải từ 1 đến 30.')
    if not (13 <= int(grid_steps) <= 101):
        raise ValueError('Mật độ lưới không hợp lệ.')
    training = result['training_rows']
    cluster = training.loc[(training['cement_type'] == str(cement_type)) &
                           (training['fly_ash_grade'] == str(fly_ash_grade))].copy()
    if len(cluster) < 3:
        raise ValueError('Chỉ đề xuất cho nhóm vật liệu có ít nhất 3 mẫu huấn luyện thực tế.')

    obs = cluster[['fly_ash_pct', 'water_binder_ratio']].to_numpy(float)
    obs_min = obs.min(axis=0)
    obs_max = obs.max(axis=0)
    bounds = [fly_ash_range, wb_range]
    mins, maxs = [], []
    for k, bound in enumerate(bounds):
        low, high = (obs_min[k], obs_max[k]) if bound is None else map(float, bound)
        if not (np.isfinite(low) and np.isfinite(high) and
                obs_min[k] - 1e-9 <= low <= high <= obs_max[k] + 1e-9):
            raise ValueError(f'Khoảng {NUM_COLS[k]} vượt miền thực nghiệm của nhóm vật liệu.')
        mins.append(low)
        maxs.append(high)

    # Observed records first. Keep their original experiment and reference info.
    obs_frame = cluster.loc[
        cluster['fly_ash_pct'].between(mins[0], maxs[0]) &
        cluster['water_binder_ratio'].between(mins[1], maxs[1])
    ].copy()
    if mode == MODES[1]:
        if fixed_wb is None or not (mins[1] <= float(fixed_wb) <= maxs[1]):
            raise ValueError('N/CKD cố định nằm ngoài khoảng đã chọn.')
        obs_frame = obs_frame.loc[np.isclose(obs_frame['water_binder_ratio'].to_numpy(float),
                                           float(fixed_wb), atol=1e-9)]
    if mode == MODES[2]:
        if fixed_fly_ash is None or not (mins[0] <= float(fixed_fly_ash) <= maxs[0]):
            raise ValueError('Tỷ lệ tro bay cố định nằm ngoài khoảng đã chọn.')
        obs_frame = obs_frame.loc[np.isclose(obs_frame['fly_ash_pct'].to_numpy(float),
                                           float(fixed_fly_ash), atol=1e-9)]

    rows = []
    for _, o in obs_frame.iterrows():
        rows.append({
            'fly_ash_pct': float(o['fly_ash_pct']),
            'water_binder_ratio': float(o['water_binder_ratio']),
            'origin': 'Mẫu thực nghiệm',
            'observed_rcm_x1e12': float(o[TARGET]) * 1e12,
            'source_reference': str(o.get('source_reference', '')),
            'source_excel_row': str(o.get('source_excel_row', '')),
            'source_group': str(o.get('source_group', '')),
            'nearest_measured_reference': '',
            'nearest_measured_excel_row': '',
            'support_distance': 0.0,
        })

    # Optional interpolation in the *observed category pair* convex hull.
    # Restrict to points within 0.80 normalized distances from a REAL training
    # mix, so the search does not roam far into unsampled material combinations.
    if include_interpolations and len(np.unique(obs, axis=0)) >= 3:
        scales = np.std(training[NUM_COLS].to_numpy(float), axis=0, ddof=0)
        scales = np.maximum(scales, np.array([1.0, 0.01]))
        centered = (obs - obs.mean(axis=0)) / scales
        if np.linalg.matrix_rank(centered, tol=1e-8) == 2:
            if mode == MODES[0]:
                x = np.linspace(mins[0], maxs[0], int(grid_steps))
                y = np.linspace(mins[1], maxs[1], int(grid_steps))
                gx, gy = np.meshgrid(x, y)
                grid = np.column_stack((gx.ravel(), gy.ravel()))
            elif mode == MODES[1]:
                grid = np.column_stack((np.linspace(mins[0], maxs[0], int(grid_steps)*3),
                                        np.full(int(grid_steps)*3, float(fixed_wb))))
            else:
                grid = np.column_stack((np.full(int(grid_steps)*3, float(fixed_fly_ash)),
                                        np.linspace(mins[1], maxs[1], int(grid_steps)*3)))
            try:
                tri = Delaunay(obs / scales, qhull_options='QJ Pp')
                inside = tri.find_simplex(grid / scales) >= 0
            except QhullError:
                inside = np.zeros(len(grid), dtype=bool)
            distances = np.linalg.norm((grid[:, None, :] - obs[None, :, :]) / scales,
                                       axis=2)
            nearest_indices = distances.argmin(axis=1)
            nearest_distances = distances.min(axis=1)
            good = inside & (nearest_distances <= 0.80)
            for point, near_idx, dist in zip(grid[good], nearest_indices[good], nearest_distances[good]):
                reference = cluster.iloc[int(near_idx)]
                rows.append({
                    'fly_ash_pct': float(point[0]),
                    'water_binder_ratio': float(point[1]),
                    'origin': 'Nội suy (chưa thí nghiệm)',
                    'observed_rcm_x1e12': np.nan,
                    'source_reference': '',
                    'source_excel_row': '',
                    'source_group': '',
                    'nearest_measured_reference': str(reference.get('source_reference', '')),
                    'nearest_measured_excel_row': str(reference.get('source_excel_row', '')),
                    'support_distance': float(dist),
                })
    if not rows:
        return {
            'status': 'no_candidates', 'shortlisted': pd.DataFrame(),
            'all_feasible': pd.DataFrame(), 'candidate_count': 0,
            'feasible_count': 0, 'measured_pass_count': 0,
            'predicted_only_count': 0, 'false_positive_count': 0,
            'target_rcm_x1e12': target,
            'screening_limit_x1e12': np.nan, 'training_count_for_pair': len(cluster),
            'notes': 'No candidates inside the requested data domain.'
        }
    candidates = pd.DataFrame(rows)
    # Do not discard repeated independent measurements of identical inputs.
    # Only remove repeated untested interpolation grid points.
    observed_mask = candidates['origin'].eq('M\u1eabu th\u1ef1c nghi\u1ec7m')
    measured_rows = candidates.loc[observed_mask].copy()
    new_rows = candidates.loc[~observed_mask].copy()
    if not new_rows.empty:
        new_rows['_fa'] = new_rows['fly_ash_pct'].round(5)
        new_rows['_wb'] = new_rows['water_binder_ratio'].round(6)
        new_rows = new_rows.drop_duplicates(['_fa', '_wb']).drop(columns=['_fa', '_wb'])
    candidates = pd.concat([measured_rows, new_rows], ignore_index=True)
    measured_mask = candidates['origin'].eq('M\u1eabu th\u1ef1c nghi\u1ec7m')

    features = candidates[['fly_ash_pct', 'water_binder_ratio']].copy()
    features['cement_type'] = str(cement_type)
    features['fly_ash_grade'] = str(fly_ash_grade)
    predictions = (10.0 ** result['model'].predict(features[FEATURES])) * 1e12
    if not np.all(np.isfinite(predictions)) or not np.all(predictions > 0):
        raise ValueError('Invalid RCM prediction returned by the model.')
    candidates['predicted_rcm_x1e12'] = predictions
    candidates['cement_type'] = str(cement_type)
    candidates['fly_ash_grade'] = str(fly_ash_grade)
    candidates['target_rcm_x1e12'] = target
    # CV error is estimated from TRAINING groups only; holdout never enters design.
    cv_row = result['table'].loc[result['table']['Thu\u1eadt to\u00e1n'] == result['selected']]
    cv_mae = float(cv_row.iloc[0]['CV MAE (\u00d710\u207b\u00b9\u00b2 m\u00b2/s)'])
    absolute_buffer = float(buffer_mae_multiple) * cv_mae
    gate = target - absolute_buffer
    if gate <= 0:
        raise ValueError('Target is below the selected safety buffer.')
    candidates['pass_predicted_limit'] = candidates['predicted_rcm_x1e12'].le(gate + 1e-12)
    candidates['difference_to_goal_x1e12'] = target - candidates['predicted_rcm_x1e12']

    # Distinguish *measured evidence* from *model-screened hypotheses*.
    # Never call an actual measured failure a pass just because the model predicts a pass.
    actual_pass = candidates['observed_rcm_x1e12'].le(target + 1e-12)
    candidates['pass_measured_limit'] = pd.Series(pd.NA, index=candidates.index, dtype='boolean')
    candidates.loc[measured_mask, 'pass_measured_limit'] = actual_pass[measured_mask].astype(bool)
    candidates['pass_screening'] = np.where(
        measured_mask, actual_pass.to_numpy(bool), candidates['pass_predicted_limit'].to_numpy(bool)
    )
    candidates['evidence_level'] = np.select([
        measured_mask & actual_pass,
        measured_mask & ~actual_pass,
        ~measured_mask & candidates['pass_predicted_limit'],
        ~measured_mask & ~candidates['pass_predicted_limit'],
    ], [
        'M\u1eabu \u0111o: RCM th\u1ef1c t\u1ebf \u0111\u1ea1t ng\u01b0\u1ee1ng',
        'M\u1eabu \u0111o: RCM th\u1ef1c t\u1ebf kh\u00f4ng \u0111\u1ea1t',
        'N\u1ed9i suy: ch\u1ec9 d\u1ef1 b\u00e1o \u0111\u1ea1t, ch\u1edd th\u1eed nghi\u1ec7m',
        'N\u1ed9i suy: d\u1ef1 b\u00e1o kh\u00f4ng \u0111\u1ea1t',
    ], default='Ch\u01b0a ph\u00e2n lo\u1ea1i')
    candidates['prediction_vs_measurement_conflict'] = (
        measured_mask &
        (candidates['predicted_rcm_x1e12'].le(target + 1e-12) != actual_pass)
    )
    # The measured value takes precedence for real samples; predictions are used
    # only to rank the unmeasured alternatives.
    candidates['ranking_rcm_x1e12'] = np.where(
        measured_mask, candidates['observed_rcm_x1e12'], candidates['predicted_rcm_x1e12']
    )
    passing = candidates.loc[candidates['pass_screening']].copy()
    status = 'candidates_found' if len(passing) else 'no_candidates_meet_limit'
    available = passing if len(passing) else candidates.copy()
    if not len(passing):
        available = available.assign(_gap=(available['ranking_rcm_x1e12'] - target).abs())
        available = available.sort_values(['_gap', 'support_distance']).drop(columns='_gap')
    elif preference == PREFERENCES[0]:
        available = available.assign(_gap=(target - available['ranking_rcm_x1e12']).abs())
        available = available.sort_values(['_gap', 'support_distance']).drop(columns='_gap')
    elif preference == PREFERENCES[1]:
        available = available.sort_values(['fly_ash_pct', 'support_distance', 'ranking_rcm_x1e12'])
    elif preference == PREFERENCES[2]:
        available = available.sort_values(['water_binder_ratio', 'support_distance', 'ranking_rcm_x1e12'])
    else:
        available = available.assign(_obs=available['origin'].ne('M\u1eabu th\u1ef1c nghi\u1ec7m').astype(int))
        available = available.sort_values(['_obs', 'support_distance', 'ranking_rcm_x1e12']).drop(columns='_obs')

    chosen = []
    for _, row in available.iterrows():
        if all(abs(row['fly_ash_pct'] - x['fly_ash_pct']) >= 1.4 or
               abs(row['water_binder_ratio'] - x['water_binder_ratio']) >= 0.013
               for x in chosen):
            chosen.append(row)
        if len(chosen) >= int(top_n):
            break
    short = pd.DataFrame(chosen, columns=candidates.columns) if chosen else candidates.head(0)
    return {
        'status': status,
        'shortlisted': short.reset_index(drop=True),
        'all_feasible': passing.reset_index(drop=True),
        'all_candidates': candidates.reset_index(drop=True),
        'candidate_count': int(len(candidates)),
        'feasible_count': int(len(passing)),  # screening hits, NOT verified designs
        'measured_pass_count': int((measured_mask & actual_pass).sum()),
        'predicted_only_count': int((~measured_mask & candidates['pass_predicted_limit']).sum()),
        'false_positive_count': int((measured_mask & candidates['prediction_vs_measurement_conflict'] &
                                     ~actual_pass).sum()),
        'target_rcm_x1e12': target,
        'screening_limit_x1e12': gate,
        'training_count_for_pair': int(len(cluster)),
        'notes': 'Measured samples are evaluated using measured values; interpolation is model screening only.',
    }
