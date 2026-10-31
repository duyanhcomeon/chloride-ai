"""Regression gates for provenance and units of the supplementary research data."""
from pathlib import Path
from research_catalog import load_research,research_summary
ROOT=Path(__file__).resolve().parents[1]

def test_sources_and_rows_counts():
    s,o=load_research(ROOT)
    r=research_summary(s,o)
    assert r['source_count']==14
    assert r['row_count']==19
    assert r['study_counts']=={'MENENDEZ_2019':10,'ZAFAR_2015':5,'PARK_2020':4}
    assert r['errors']==[]

def test_no_fictitious_structure_lifetime_training():
    s,o=load_research(ROOT)
    r=research_summary(s,o)
    assert not r['can_train_structure_lifespan']
    assert all(float(x['time_original'])>0 for x in o)
    assert all(x['event_observed']=='1' for x in o)

def test_readme_contains_source_links():
    s,_=load_research(ROOT)
    doc=(ROOT/'README_NGHIEN_CUU_v09.md').read_text(encoding='utf8')
    assert all(record['url'] in doc for record in s)
