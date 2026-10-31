"""Regression tests for the ML-only workflow, not for engineering design."""
from pathlib import Path

import pandas as pd
from rcm_training import clean, train_compare, predict_rcm, MODELS

ROOT=Path(__file__).resolve().parents[1]

def test_real_dataset_and_groups():
    df=clean(pd.read_csv(ROOT/'real_rcm_flyash_215.csv'))
    assert len(df)==215
    assert df.source_group.nunique()>=90
    assert df.d_rcm_28_m2_s.gt(0).all()

def test_nine_models_and_holdout():
    df=pd.read_csv(ROOT/'real_rcm_flyash_215.csv')
    out=train_compare(df)
    assert len(out['table'])==9
    assert out['train_count']+out['test_count']==215
    assert out['test_count']==39
    assert out['best_cv'] in MODELS
    assert out['holdout_scores']['MAE_x1e12']>=0

def test_in_range_prediction():
    df=pd.read_csv(ROOT/'real_rcm_flyash_215.csv')
    out=train_compare(df)
    prediction=predict_rcm(out,25,.40,'PO425','1.0')
    assert 0<prediction<1e-9

def test_app_does_not_import_fick():
    app=(ROOT/'app.py').read_text(encoding='utf-8')
    for forbidden in ('from physics import','from corrosion import','initiation_years(', 'serviceability_indicator(', 'from comparison import'):
        assert forbidden not in app
