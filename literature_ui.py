"""Visible research library; physically segregated from RCM and lifetime ML."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st
from research_catalog import SOURCES, INITIATION, load_research, research_summary


def render_literature_tab(root: Path):
    root = Path(root)
    st.subheader('Thư viện nghiên cứu về thời gian khởi phát ăn mòn')
    st.write('Tổng hợp số liệu từ các công bố về bê tông có tro bay và môi trường ion clorua. '
             '**Tách biệt hoàn toàn khỏi mô hình RCM hiện tại và tab Tuổi thọ AI**.')
    try:
        src, obs = load_research(root)
    except (OSError, ValueError) as e:
        st.error('Thiếu tệp nguồn nghiên cứu trong gói phát hành: ' + str(e))
        return
    info = research_summary(src, obs)
    if info['errors']:
        st.error('Bảng dữ liệu chưa vượt kiểm tra nguồn và đơn vị: ' + '; '.join(info['errors'][:4]))
        return
    c1,c2,c3,c4=st.columns(4)
    c1.metric('Tài liệu đã tổng hợp', info['source_count'])
    c2.metric('Dòng kết quả trích xuất', info['row_count'])
    c3.metric('Nghiên cứu có dữ liệu thời gian', info['group_count'])
    c4.metric('Nguồn đã đối chiếu liên kết', info['verified_links'])
    st.warning('**19 dòng là các cấp phối/giá trị trung bình đã công bố, KHÔNG phải 19 mẫu độc lập.** '
               'Các thí nghiệm tăng tốc, ngâm muối và phơi nhiễm dài hạn không cùng điều kiện. '
               'Quy đổi giờ thành ngày chỉ để đọc bảng, không quy đổi thành năm tuổi thọ sử dụng.')
    st.markdown('#### 1. Kết quả thời gian khởi phát ăn mòn đã trích xuất')
    raw=pd.DataFrame(obs)
    for name in ['fly_ash_pct','bottom_ash_pct','slag_pct','water_binder_ratio','cover_mm',
                 'time_original','time_to_corrosion_days']:
        raw[name] = pd.to_numeric(raw[name], errors='coerce')
    study = st.selectbox('Chọn nghiên cứu để xem riêng (không gộp giao thức)',
                         ['Tất cả — chỉ xem bảng'] + sorted(info['study_counts']),
                         key='literature_study')
    show=raw if study.startswith('Tất cả') else raw.loc[raw['study_id'].eq(study)]
    st.dataframe(show[['study_id','mix_id','fly_ash_pct','water_binder_ratio','cover_mm',
                        'time_original','time_unit','time_to_corrosion_days','record_level',
                        'source_url']],use_container_width=True,hide_index=True,height=360)
    if not study.startswith('Tất cả'):
        chart=alt.Chart(show).mark_bar(cornerRadiusEnd=5,color='#1a7b89').encode(
            x=alt.X('mix_id:N',sort=None,title='Cấp phối theo nghiên cứu gốc'),
            y=alt.Y('time_to_corrosion_days:Q',title='Thời gian quan sát (ngày)'),
            tooltip=['mix_id:N','fly_ash_pct:Q','time_original:Q','time_unit:N',
                     alt.Tooltip('time_to_corrosion_days:Q',format='.3f')]
        ).properties(height=310)
        st.altair_chart(chart,use_container_width=True)
        st.caption('Biểu đồ chỉ gồm các cấp phối trong cùng nghiên cứu. Không dùng để dự báo tuổi thọ công trình.')
    st.download_button('⬇ Tải 19 dòng có nguồn và phương pháp thử',
        (root/INITIATION).read_bytes(),file_name=INITIATION,mime='text/csv',key='download_research19')
    st.markdown('#### 2. Danh mục các nghiên cứu, bài báo và bộ dữ liệu')
    kinds=['Tất cả']+sorted({s['evidence_type'] for s in src})
    category=st.selectbox('Lọc theo dạng bằng chứng',kinds,key='research_kind')
    subset=[s for s in src if category=='Tất cả' or s['evidence_type']==category]
    for s in subset:
        with st.expander(f"{s['source_id']} · {s['year']} · {s['authors']} · {s['topic']}"):
            st.write('**Dạng dữ liệu:** '+s['evidence_type'])
            st.write('**Phạm vi:** '+s['coverage'])
            st.write('**Đại lượng:** '+s['variables'])
            st.write('**Trạng thái:** '+s['availability'])
            st.caption(s['caution'])
            st.caption('Đối chiếu nguồn: '+s['verification_status'])
            st.markdown('[Mở bài báo / trang dữ liệu gốc]('+s['url']+')')
    st.download_button('⬇ Tải 14 nguồn và đường dẫn DOI',
        (root/SOURCES).read_bytes(),file_name=SOURCES,mime='text/csv',key='download_references14')
    st.markdown('#### 3. Khả năng bổ sung vào mô hình AI tuổi thọ')
    st.error('**Chưa đủ điều kiện để huấn luyện và xuất số năm tuổi thọ thực tế.** '+info['reason'])
    st.info('Muốn học thời gian khởi phát bằng AI: cần dữ liệu theo **từng mẫu thí nghiệm**, '
            'ngày xảy ra sự kiện / ngày kết thúc theo dõi và trạng thái **censored**, '
            'điều kiện phơi nhiễm, chiều bảo vệ, cấp phối, cùng mô tả giao thức. '
            'Mô hình sinh tồn (survival analysis) phù hợp khi nhiều mẫu chưa xảy ra ăn mòn.')
    st.caption('Chỉ tiêu RCM/D28, thời gian khởi phát ăn mòn, nứt và mất khả năng chịu lực là các nhãn khác nhau. '
               'Nguồn DOI hiển thị trong CSV, chưa kiểm tra đầy đủ nội dung của mọi bài báo.')
