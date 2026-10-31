"""Scientific source catalogue: lookup and provenance checks only, no training.

The 19 observations are *mixture-level or published means* measured with
heterogeneous protocols, never 19 independent service-life observations.
"""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path

SOURCES = 'research_sources_14.csv'
INITIATION = 'corrosion_initiation_19_groups.csv'


def _read(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_research(root: Path):
    root = Path(root)
    sources, observed = _read(root / SOURCES), _read(root / INITIATION)
    return sources, observed


def research_summary(sources, rows):
    studies = Counter(row['study_id'] for row in rows)
    errors = []
    if len({s['source_id'] for s in sources}) != len(sources):
        errors.append('Mã nguồn trùng lặp.')
    links = {s['url'] for s in sources}
    for i, row in enumerate(rows, 2):
        if row['source_url'] not in links:
            errors.append(f'Dòng {i}: thiếu DOI trong thư viện nguồn.')
        try:
            t = float(row['time_original'])
            days = float(row['time_to_corrosion_days'])
            expected = t / 24 if row['time_unit'].strip() == 'h' else t
            if abs(days - expected) > .001:
                errors.append(f'Dòng {i}: thời gian quy đổi không khớp.')
            if t <= 0 or days <= 0:
                errors.append(f'Dòng {i}: thời gian không dương.')
        except (ValueError, KeyError, TypeError):
            errors.append(f'Dòng {i}: thời gian không hợp lệ.')
        if row.get('event_observed') != '1':
            errors.append(f'Dòng {i}: không phải thời gian sự kiện đã quan sát.')
    return {
        'source_count':len(sources),
        'row_count':len(rows),
        'study_counts':dict(studies),
        'group_count':len(studies),
        'verified_links':sum(s['verification_status'].startswith('Đã') for s in sources),
        'errors':errors,
        'can_train_structure_lifespan':False,
        'reason':'Chỉ 3 nhóm nghiên cứu, giao thức khác nhau, có giá trị trung bình nhóm; thiếu nhãn tuổi thọ kết cấu, RCM tương ứng và biến môi trường đồng nhất.',
    }
