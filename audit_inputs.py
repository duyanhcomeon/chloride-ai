"""Kiểm tra CSV CloriLife, chỉ kiểm tra cấu trúc — KHÔNG tạo nhãn AI."""
from __future__ import annotations
import argparse, json
import pandas as pd
RCM_COLUMNS={'fly_ash_pct','water_binder_ratio','cement_type','fly_ash_grade','d_rcm_28_m2_s','source_group'}
SURV_COLUMNS={'fly_ash_pct','water_binder_ratio','cover_mm','exposure_class','observed_duration_years','event_observed','source_group','limit_state','exposure_protocol','source_reference'}

def audit_rcm(data:pd.DataFrame)->dict:
    missing=sorted(RCM_COLUMNS-set(data.columns))
    report={'type':'RCM','rows':len(data),'missing_columns':missing,'ready_to_attempt_training':False}
    if missing:return report
    report['source_groups']=int(data['source_group'].nunique())
    report['missing_values']=data[list(RCM_COLUMNS)].isna().sum().astype(int).to_dict()
    vals=pd.to_numeric(data['d_rcm_28_m2_s'],errors='coerce')
    report['invalid_rcm_rows']=int((vals.isna() | (vals<=0)).sum())
    report['ready_to_attempt_training']=(len(data)>=30 and report['source_groups']>=5 and report['invalid_rcm_rows']==0 and not data['source_group'].isna().any())
    return report

def audit_survival(data:pd.DataFrame)->dict:
    missing=sorted(SURV_COLUMNS-set(data.columns))
    report={'type':'SURVIVAL','rows':len(data),'missing_columns':missing,'ready_to_attempt_training':False}
    if missing:return report
    events=pd.to_numeric(data['event_observed'],errors='coerce')
    durations=pd.to_numeric(data['observed_duration_years'],errors='coerce')
    report['invalid_event_rows']=int((~events.isin([0,1])).sum())
    report['invalid_duration_rows']=int((durations.isna() | (durations<=0)).sum())
    report['cohorts']=[]
    for (state,protocol), group in data.groupby(['limit_state','exposure_protocol'],dropna=False):
        e=pd.to_numeric(group['event_observed'],errors='coerce')
        g={'limit_state':str(state),'protocol':str(protocol),'rows':len(group),'observed_events':int(e.eq(1).sum()),'independent_sources':int(group['source_group'].nunique())}
        g['eligible_counts']=(g['rows']>=40 and g['observed_events']>=12 and g['independent_sources']>=8)
        report['cohorts'].append(g)
    report['ready_to_attempt_training']=(report['invalid_event_rows']==0 and report['invalid_duration_rows']==0 and any(g['eligible_counts'] for g in report['cohorts']) and not data[list(SURV_COLUMNS)].isna().any().any())
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='CloriLife CSV structural audit. No data transformation or label creation.')
    ap.add_argument('--type',choices=['rcm','survival'],required=True)
    ap.add_argument('--file',required=True)
    args=ap.parse_args()
    data=pd.read_csv(args.file,encoding='utf-8-sig')
    report=audit_rcm(data) if args.type=='rcm' else audit_survival(data)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    print('LƯU Ý: điều kiện cần, KHÔNG phải xác thực khoa học hoặc chứng nhận đủ huấn luyện.')
