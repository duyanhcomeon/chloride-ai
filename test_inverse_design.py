"""End-to-end inverse design tests against the actual 215-row RCM dataset.

The inverse algorithm must never use holdout experimental rows to propose mixes.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from rcm_training import train_compare, clean, predict_rcm
from inverse_design import training_pairs, inverse_search, MODES, PREFERENCES

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(scope='module')
def fitted():
    data=clean(pd.read_csv(ROOT/'real_rcm_flyash_215.csv'))
    return train_compare(data)

@pytest.fixture(scope='module')
def materials(fitted):
    most=training_pairs(fitted).iloc[0]
    return str(most['cement_type']), str(most['fly_ash_grade'])


def test_uses_only_training_records(fitted):
    observed_sources=set(fitted['training_rows']['source_group'].astype(str))
    assert len(fitted['training_rows'])==fitted['train_count']
    assert observed_sources.isdisjoint(set(fitted['holdout']['source_group'].astype(str)))


def test_only_actually_observed_material_pairs(fitted):
    pairs=training_pairs(fitted)
    assert len(pairs)>=5
    assert pairs.n_training_rows.ge(3).all()
    assert all(
        len(fitted['training_rows'].query('cement_type == @c and fly_ash_grade == @g'))>=3
        for c,g in zip(pairs.cement_type,pairs.fly_ash_grade)
    )


def test_inverse_returns_multiple_feasible_and_forward_consistent(fitted,materials):
    c,g=materials
    out=inverse_search(fitted, target_rcm_x1e12=15.0,cement_type=c,fly_ash_grade=g)
    assert out['status']=='candidates_found'
    assert out['feasible_count']>5
    assert 1<=len(out['shortlisted'])<=8
    assert out['shortlisted']['pass_screening'].all()
    observed=out['shortlisted']['observed_rcm_x1e12'].notna()
    assert (out['shortlisted'].loc[observed,'observed_rcm_x1e12']<=15.0+1e-10).all()
    assert (out['shortlisted'].loc[~observed,'predicted_rcm_x1e12']<=15.0+1e-10).all()
    for _,r in out['shortlisted'].head(3).iterrows():
        direct=predict_rcm(fitted,float(r.fly_ash_pct),float(r.water_binder_ratio),c,g)*1e12
        assert np.isclose(direct,float(r.predicted_rcm_x1e12),rtol=1e-12)


def test_untestable_near_misses_not_labelled_feasible(fitted,materials):
    c,g=materials
    out=inverse_search(fitted,0.2,c,g)
    assert out['status']=='no_candidates_meet_limit'
    assert out['feasible_count']==0
    assert len(out['shortlisted'])>=1
    assert not out['shortlisted']['pass_predicted_limit'].any()


def test_observed_only_is_strict(fitted,materials):
    c,g=materials
    out=inverse_search(fitted,20,c,g,include_interpolations=False)
    assert (out['shortlisted']['origin']=='Mẫu thực nghiệm').all()
    assert out['all_feasible']['observed_rcm_x1e12'].notna().all()
    assert set(out['all_feasible']['source_group']).issubset(set(fitted['training_rows']['source_group']))


def test_interpolated_solutions_have_missing_measured_rcm(fitted,materials):
    c,g=materials
    out=inverse_search(fitted,20,c,g,include_interpolations=True,preference=PREFERENCES[0])
    interp=out['all_feasible'].loc[out['all_feasible']['origin'].str.contains('Nội suy')]
    assert len(interp)>0
    assert interp['observed_rcm_x1e12'].isna().all()
    assert interp['support_distance'].le(0.80+1e-9).all()


def test_fixed_water_binder_solves_only_flyash(fitted,materials):
    c,g=materials
    frame=fitted['training_rows'].loc[
        (fitted['training_rows']['cement_type']==c)&
        (fitted['training_rows']['fly_ash_grade']==g)]
    wb=float(frame['water_binder_ratio'].median())
    out=inverse_search(fitted,20,c,g,mode=MODES[1],fixed_wb=wb)
    if out['candidate_count']:
        assert np.allclose(out['shortlisted']['water_binder_ratio'],wb)


def test_fixed_flyash_solves_only_water_binder(fitted,materials):
    c,g=materials
    frame=fitted['training_rows'].loc[
        (fitted['training_rows']['cement_type']==c)&
        (fitted['training_rows']['fly_ash_grade']==g)]
    fa=float(frame['fly_ash_pct'].median())
    out=inverse_search(fitted,20,c,g,mode=MODES[2],fixed_fly_ash=fa)
    if out['candidate_count']:
        assert np.allclose(out['shortlisted']['fly_ash_pct'],fa)


def test_safety_buffer_reduces_allowable_threshold(fitted,materials):
    c,g=materials
    out=inverse_search(fitted,16,c,g,buffer_mae_multiple=1.0)
    assert np.isclose(out['screening_limit_x1e12'],
                      16-float(fitted['table'].loc[fitted['table']['Thuật toán'] == fitted['selected'], 'CV MAE (×10⁻¹² m²/s)'].iloc[0]))
    r=out['all_feasible']
    measured=r['observed_rcm_x1e12'].notna()
    assert (r.loc[measured,'observed_rcm_x1e12']<=16.0+1e-10).all()
    assert (r.loc[~measured,'predicted_rcm_x1e12']<=out['screening_limit_x1e12']+1e-10).all()

@pytest.mark.parametrize('bad_target',[0,-1,np.nan,np.inf])
def test_reject_invalid_target(fitted,materials,bad_target):
    c,g=materials
    with pytest.raises(ValueError):inverse_search(fitted,bad_target,c,g)


def test_reject_extrapolation(fitted,materials):
    c,g=materials
    with pytest.raises(ValueError):inverse_search(fitted,8,c,g,fly_ash_range=(-1,80))


def test_reject_unseen_categorical_combinations(fitted):
    with pytest.raises(ValueError):inverse_search(fitted,8,'NOT_REPORTED','1.0')


def test_inverse_app_does_not_use_fick():
    app=(ROOT/'app.py').read_text(encoding='utf-8')
    assert 'with inverse_tab:' in app
    assert 'from physics import' not in app
    assert 'from corrosion import' not in app


class _AlwaysPredictLow:
    def predict(self, features):
        return np.full(len(features), np.log10(7e-12))


def _stub_fitted(measured):
    # All three are measured samples; the model predicts 7 for every sample.
    rows=[]
    for ix, val in enumerate(measured):
        rows.append(dict(fly_ash_pct=10+ix*5, water_binder_ratio=0.35+0.03*ix,
                         cement_type='P', fly_ash_grade='1.0', d_rcm_28_m2_s=val*1e-12,
                         source_reference=f'Paper {ix}', source_excel_row=ix+2,
                         source_group=f'paper{ix}'))
    return dict(training_rows=pd.DataFrame(rows), model=_AlwaysPredictLow(),
                table=pd.DataFrame([{'Thu\u1eadt to\u00e1n':'Fake',
                     'CV MAE (\u00d710\u207b\u00b9\u00b2 m\u00b2/s)':1.5}]),
                selected='Fake')


def test_measured_fail_cannot_be_saved_by_model_prediction():
    result=_stub_fitted([12.0, 6.0, 10.0])
    out=inverse_search(result,8.0,'P','1.0',include_interpolations=False)
    assert out['false_positive_count']==2
    assert out['measured_pass_count']==1
    assert out['predicted_only_count']==0
    assert out['feasible_count']==1
    assert out['all_feasible']['observed_rcm_x1e12'].tolist()==[6.0]
    assert not out['all_candidates'].loc[
        out['all_candidates']['observed_rcm_x1e12']>8.0,'pass_screening'].any()


def test_no_measured_pass_despite_all_predicted_pass():
    result=_stub_fitted([12.0, 13.0, 10.0])
    out=inverse_search(result,8.0,'P','1.0',include_interpolations=False)
    assert out['status']=='no_candidates_meet_limit'
    assert out['feasible_count']==0
    assert out['false_positive_count']==3
    assert not out['shortlisted']['pass_screening'].any()


def test_app_does_not_use_global_streamlit_stop():
    import ast
    code=(ROOT/'app.py').read_text(encoding='utf-8')
    ast.parse(code)
    assert 'st.stop()' not in code
    assert "if result is None:" in code
    assert "'evidence_level'" in code
