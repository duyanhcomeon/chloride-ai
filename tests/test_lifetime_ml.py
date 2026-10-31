"""Unit tests only; generated fixtures are NOT scientific lifespan observations."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lifetime_ml import (
    AUTO, MODELS, LifetimeDataError, REQUIRED_COLUMNS, available_states,
    clean_lifetime, predict_lifetime, schema_csv_bytes, train_lifetime,
)


def fabricated_test_fixture(n_sources=12, n_each=6):
    """FICTITIOUS values exclusively to test software functions, not training data."""
    rng = np.random.default_rng(14)
    rows = []
    for g in range(n_sources):
        for j in range(n_each):
            fly = rng.uniform(8., 35.)
            wb = rng.uniform(.30, .55)
            rcm = rng.uniform(4., 12.)
            cover = rng.uniform(25., 75.)
            env = ['marine','urban','splash'][g % 3]
            years = max(.1, 30 + .42*fly - 25*wb - .7*rcm + .35*cover
                        - (13 if env=='splash' else 0) + rng.normal(0, 1.5))
            rows.append({'fly_ash_pct':fly,'water_binder_ratio':wb,
                         'd_rcm_28_m2_s':rcm*1e-12,'cover_mm':cover,
                         'exposure_class':env,'time_to_limit_years':years,
                         'limit_state':'first_crack_0p3mm','event_observed':1,
                         'source_group':f'test_{g}',
                         'source_reference':f'FAKE TEST SOURCE {g}'})
    return pd.DataFrame(rows)


def test_schema_no_fabricated_data():
    template = schema_csv_bytes().decode('utf-8-sig').strip().splitlines()
    assert len(template) == 1
    assert template[0] == ','.join(REQUIRED_COLUMNS)
    assert 'time_to_limit_years' in template[0]


def test_no_life_labels_in_rcm_data():
    df = pd.read_csv('real_rcm_flyash_215.csv')
    with pytest.raises(LifetimeDataError,match='Thiếu cột'):
        clean_lifetime(df,'first_crack_0p3mm')


def test_censored_is_not_treated_as_actual_failure():
    frame = fabricated_test_fixture()
    frame.loc[0,'event_observed']=0
    with pytest.raises(LifetimeDataError,match='censor'):
        clean_lifetime(frame,'first_crack_0p3mm')


def test_no_source_reference_is_rejected():
    frame = fabricated_test_fixture()
    frame.loc[0,'source_reference']=None
    with pytest.raises(LifetimeDataError,match='nguồn'):
        clean_lifetime(frame,'first_crack_0p3mm')


def test_wrong_label_missing_rejected():
    frame = fabricated_test_fixture()
    frame.loc[1,'time_to_limit_years']=np.nan
    with pytest.raises(LifetimeDataError,match='dòng thiếu'):
        clean_lifetime(frame,'first_crack_0p3mm')


def test_state_separation():
    frame = fabricated_test_fixture()
    alt = frame.iloc[:2].copy()
    alt['limit_state']='failure_in_bending'
    assert available_states(pd.concat([frame,alt])) == [
        'failure_in_bending','first_crack_0p3mm'
    ]
    df = clean_lifetime(pd.concat([frame,alt]),'first_crack_0p3mm')
    assert len(df)==len(frame)


def test_too_little_data_rejected():
    frame = fabricated_test_fixture(n_sources=7,n_each=3)
    with pytest.raises(LifetimeDataError,match='40'):
        clean_lifetime(frame,'first_crack_0p3mm')


def test_model_training_cv_and_no_group_leakage():
    frame = fabricated_test_fixture()
    r=train_lifetime(frame,'first_crack_0p3mm',selection=AUTO)
    assert len(r['table'])==len(MODELS)==9
    assert set(r['train_source_set']).isdisjoint(set(r['test_source_set']))
    assert r['total_count']==len(frame)
    assert len(r['holdout'])==r['test_count']
    assert np.isfinite(r['table']['CV MAE (năm)']).all()
    assert r['scores']['MAE_years']>=0
    midpoint={col:np.mean(r['bounds'][col]) for col in r['bounds']}
    prediction=predict_lifetime(r,fly_ash_pct=midpoint['fly_ash_pct'],
        water_binder_ratio=midpoint['water_binder_ratio'],
        d_rcm_28_m2_s=midpoint['rcm_x1e12']*1e-12,
        cover_mm=midpoint['cover_mm'],
        exposure_class=r['exposures'][0])
    assert np.isfinite(prediction) and prediction>0
    with pytest.raises(LifetimeDataError,match='ngoài miền'):
        predict_lifetime(r,fly_ash_pct=-10,water_binder_ratio=.40,
            d_rcm_28_m2_s=8e-12,cover_mm=50,
            exposure_class=r['exposures'][0])
    with pytest.raises(LifetimeDataError,match='Môi trường'):
        predict_lifetime(r,fly_ash_pct=midpoint['fly_ash_pct'],
            water_binder_ratio=midpoint['water_binder_ratio'],
            d_rcm_28_m2_s=midpoint['rcm_x1e12']*1e-12,
            cover_mm=midpoint['cover_mm'], exposure_class='extra_new_environment')


def test_choice_specific_model():
    frame=fabricated_test_fixture()
    r=train_lifetime(frame,'first_crack_0p3mm',selection='Ridge Regression')
    assert r['selected']=='Ridge Regression'
    assert r['scores']['MAE_years']>=0
