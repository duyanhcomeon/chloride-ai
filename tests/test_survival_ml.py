"""Synthetic fixtures for code tests ONLY; never ship as study results."""
import numpy as np
import pandas as pd
import pytest

from survival_ml import (
    AUTO, MODELS, SurvivalDataError, schema_csv_bytes, clean_survival, available_cohorts,
    comparable_concordance, ipcw_brier, train_survival, predict_survival,
    median_time, inverse_survival,
)


def fake_cases(n_studies=12, each=9, seed=6):
    rng = np.random.default_rng(seed)
    out = []
    for g in range(n_studies):
        for j in range(each):
            fa = rng.uniform(10, 45)
            wb = rng.uniform(.33, .65)
            cover = rng.uniform(26, 67)
            rcm = rng.uniform(6, 15) * 1e-12
            score = 0.012 * fa - 2. * wb + .012 * cover - .025 * (rcm*1e12)
            event_time = rng.exponential(14 * np.exp(score)) + .5
            last = rng.uniform(2, 27)
            t = min(event_time, last)
            e = int(event_time <= last)
            out.append(dict(fly_ash_pct=fa, water_binder_ratio=wb, cover_mm=cover,
                            d_rcm_28_m2_s=rcm,
                            exposure_class='marine' if g % 2 == 0 else 'splash',
                            observed_duration_years=t, event_observed=e,
                            source_group=f'test_lab_{g}', limit_state='corrosion_initiation',
                            exposure_protocol='natural_marine',
                            source_reference=f'SYNTHETIC_UNIT_TEST_{g}'))
    return pd.DataFrame(out)


def test_schema_only_header():
    schema = schema_csv_bytes().decode('utf-8-sig').splitlines()
    assert len(schema) == 1
    assert 'observed_duration_years' in schema[0]
    assert 'exposure_protocol' in schema[0]


def test_no_faked_rcm_labels():
    df = pd.read_csv('real_rcm_flyash_215.csv')
    with pytest.raises(SurvivalDataError, match='Thiếu cột'):
        clean_survival(df, 'corrosion_initiation', 'natural_marine')


def test_censored_kept():
    df = fake_cases()
    clean = clean_survival(df, 'corrosion_initiation', 'natural_marine')
    assert 0 in clean.event_observed.values and 1 in clean.event_observed.values
    assert len(clean) == len(df)


def test_provenance_and_protocol():
    df = fake_cases()
    df.loc[3, 'source_reference'] = ''
    with pytest.raises(SurvivalDataError, match='nguồn'):
        clean_survival(df, 'corrosion_initiation', 'natural_marine')
    df = fake_cases()
    accel = df.iloc[:2].copy()
    accel['exposure_protocol'] = 'accelerated_electrical'
    assert available_cohorts(pd.concat([df, accel])) == [
        ('corrosion_initiation', 'accelerated_electrical'),
        ('corrosion_initiation', 'natural_marine'),
    ]
    only_natural = clean_survival(pd.concat([df, accel]), 'corrosion_initiation', 'natural_marine')
    assert len(only_natural) == len(df)


def test_small_research_dataset_not_accepted():
    df = fake_cases(3, 7)
    with pytest.raises(SurvivalDataError, match='Chưa đủ dữ liệu'):
        clean_survival(df, 'corrosion_initiation', 'natural_marine')


def test_concordance_ties_and_censor():
    c, p = comparable_concordance([1, 2, 3], [1, 0, 1], [3, 2, 1])
    assert p == 2 and c == 1
    c, p = comparable_concordance([1, 2, 3], [0, 1, 0], [3, 2, 1])
    assert p == 1 and c == 1


def test_brier_with_censor():
    score = ipcw_brier([1, 2, 4, 5], [1, 0, 1, 0],
                       [.1, .8, .8, .8], 3,
                       [2, 3, 5, 6, 7], [1, 0, 1, 1, 0])
    assert np.isfinite(score) and score >= 0


def test_fit_reproducible_and_source_holdout():
    df = fake_cases()
    fitted = train_survival(df, 'corrosion_initiation', 'natural_marine', AUTO)
    assert fitted['model_uses_rcm']
    assert fitted['selected'] in MODELS
    assert len(fitted['table']) == 3
    assert fitted['censored_count'] > 0
    assert fitted['train_source_set'].isdisjoint(fitted['test_source_set'])
    assert fitted['evaluation']['comparable_pairs'] > 0
    assert np.isfinite(fitted['evaluation']['C_index'])
    assert fitted['model'].probabilities(df.head(2), np.array([3, 8, 15])).shape == (2,3)
    p = fitted['model'].probabilities(df.head(1), np.array([0.001, 1, 3, 8, 15]))[0]
    assert all(p >= 0) and all(p <= 1) and np.all(np.diff(p) <= 1e-8)
    if fitted['pass_quality']:
        sample = df.head(1)
        hz = min(8, fitted['max_years'])
        pred = predict_survival(fitted, sample, [hz])
        assert len(pred) == 1 and 0 <= pred.survival_probability.iloc[0] <= 1
        assert len(median_time(fitted, sample)) == 1
        with pytest.raises(SurvivalDataError, match='miền'):
            predict_survival(fitted, sample.assign(fly_ash_pct=1000), [hz])
    else:
        with pytest.raises(SurvivalDataError, match='chất lượng'):
            predict_survival(fitted, df.head(1), [3])


def test_optional_rcm_missing_does_not_fabricate():
    df = fake_cases()
    df.loc[0, 'd_rcm_28_m2_s'] = np.nan
    trained = train_survival(df, 'corrosion_initiation', 'natural_marine', 'Cox PH (Ridge)')
    assert trained['model_uses_rcm'] is False
    assert 'rcm_x1e12' not in trained['bounds']


def test_strict_no_extrapolation_even_good_model():
    df = fake_cases()
    trained = train_survival(df, 'corrosion_initiation', 'natural_marine', 'Cox PH (Ridge)')
    trained['pass_quality'] = True  # TEST ONLY: bypass holdout gate to test domain guards
    with pytest.raises(SurvivalDataError, match='Chỉ hỗ trợ'):
        predict_survival(trained, df.head(1), [trained['max_years'] + 10])
    with pytest.raises(SurvivalDataError, match='miền'):
        predict_survival(trained, df.head(1).assign(cover_mm=10000), [2])
    with pytest.raises(SurvivalDataError, match='RCM'):
        inverse_survival(trained, target_years=5, minimum_survival=.8,
                         cover_mm=45, exposure_class=trained['exposures'][0])

@pytest.mark.parametrize('algo', MODELS)
def test_each_survival_model_and_inverse_on_test_only_synthetic(algo):
    df = fake_cases()
    model = train_survival(df, 'corrosion_initiation', 'natural_marine', algo)
    # All methods must return finite monotone survival curves when calibrated/fitted.
    x = df.head(1)
    probability = model['model'].probabilities(x, np.array([0.1, 1, 4, 7, 12]))[0]
    assert np.isfinite(probability).all()
    assert np.all(np.diff(probability) <= 1e-8)
    assert np.all((probability >= 0) & (probability <= 1))
    # This bypass is limited to a unit-test fixture; not a research prediction.
    model['pass_quality'] = True
    env = model['exposures'][0]
    cover = float(np.mean(model['bounds']['cover_mm']))
    def synthetic_rcm_for_test_only(fa, wb):
        # A placeholder is intentionally used only inside the test code.
        return float(np.mean(model['bounds']['rcm_x1e12']) * 1e-12)
    options = inverse_survival(model, target_years=min(model['max_years']/2, 10),
                               minimum_survival=.2, cover_mm=cover,
                               exposure_class=env,
                               rcm_predictor=synthetic_rcm_for_test_only)
    assert len(options) >= 1 and len(options) <= 12
    assert set(options['trang_thai']) == {'THAM_KHAO_NGHIEN_CUU_KHONG_PHAI_THIET_KE'}
