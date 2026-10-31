import pandas as pd
from audit_inputs import audit_rcm,audit_survival

def test_builtin_rcm_has_studies():
    df=pd.read_csv('real_rcm_flyash_215.csv')
    result=audit_rcm(df)
    assert result['rows']==215 and result['source_groups']>8
    assert result['ready_to_attempt_training']

def test_empty_survival_template_cannot_train():
    df=pd.read_csv('CloriLife_v10_MAU_DU_LIEU_SURVIVAL.csv')
    assert not audit_survival(df)['ready_to_attempt_training']

def test_no_made_up_event_labels():
    data=pd.DataFrame({'fly_ash_pct':[20],'water_binder_ratio':[.4],'cover_mm':[30],
      'exposure_class':['XS3'],'observed_duration_years':[5], 'event_observed':[7],
      'source_group':['s1'],'limit_state':['corrosion_initiation'],
      'exposure_protocol':['natural_marine'],'source_reference':['example']})
    assert audit_survival(data)['invalid_event_rows']==1
