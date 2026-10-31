"""CloriLife AI v0.8 — forward/inverse RCM ML + conditional time-to-limit AI.

No Fick diffusion is used. The service-life module is trained ONLY on a separately
uploaded verified time-to-event dataset; the 215-row RCM dataset has no time labels.
"""
from __future__ import annotations

import base64
import hashlib
from html import escape
from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from inverse_design import MODES, PREFERENCES, inverse_search, training_pairs
from lifespan_ui import render_lifespan_tab

from rcm_training import (
    AUTO,
    MODELS,
    clean,
    predict_rcm,
    train_compare,
)

ROOT = Path(__file__).resolve().parent
SOURCE_URL = 'https://doi.org/10.5061/dryad.66t1g1k6j'
DATASET = ROOT / 'real_rcm_flyash_215.csv'
UNIT = '×10⁻¹² m²/s'

st.set_page_config(
    page_title='CloriLife AI | Học máy với dữ liệu RCM thực nghiệm',
    page_icon='🧪',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown('''
<style>
:root { --navy: #142c43; --teal: #0e887f; }
[data-testid="stAppViewContainer"] { background: #f5f7f9; }
[data-testid="stSidebar"] {background: rgba(236,242,245,.96);border-right:1px solid #dce5e9;}
.main .block-container {max-width: 1340px; padding-top:1.6rem;}
.cl-hero { border-radius: 20px; padding: 28px 34px; color:#fff;
  background:linear-gradient(120deg,#153049,#1e4d60 69%,#08796d);
  box-shadow:0 8px 26px rgba(23,52,67,.12); margin-bottom: 18px; }
.cl-kicker {font-size:11px;letter-spacing:1.8px;color:#9fe7d7;font-weight:800;}
.cl-title {font-weight:900;font-size:clamp(30px,3.3vw,49px);line-height:1.1;margin:8px 0 10px;}
.cl-title b {color:#8ce8d8;}
.cl-sub {color:#dcebf0;max-width:890px;line-height:1.6;font-size:14px;}
.cl-pills {display:flex;gap:9px;flex-wrap:wrap;margin-top:19px;}
.cl-pill {border:1px solid #537d89;color:#e7fffa;border-radius:30px;
 font-size:11px;font-weight:700;padding:5px 12px;}
.cl-caption {color:#4b6a7a;font-size:13px;margin:2px 0 15px;}
.cl-foot {margin-top:28px;text-align:center;color:#677f8d;font-size:11px;}
[data-testid="stMetric"] {border:1px solid #dce6eb;border-radius:13px;
 background:rgba(255,255,255,.92);padding:12px 14px;}
[data-testid="stMetricValue"] {color:#17546a;}
</style>
''', unsafe_allow_html=True)
texture = ROOT / 'concrete_texture.jpg'
if texture.is_file():
    b64 = base64.b64encode(texture.read_bytes()).decode('ascii')
    st.markdown(
        "<style>[data-testid='stAppViewContainer']{background:"
        "linear-gradient(rgba(248,250,252,.89),rgba(248,250,252,.93)),"
        f"url('data:image/jpeg;base64,{b64}') center/cover fixed!important;}}"
        "</style>",
        unsafe_allow_html=True,
    )

@st.cache_resource(show_spinner='Huấn luyện 9 mô hình và kiểm định chéo theo nguồn…')
def cached_train(csv_bytes: bytes, chosen_name: str):
    data = pd.read_csv(BytesIO(csv_bytes))
    return train_compare(data, selection=chosen_name)

@st.cache_data(show_spinner=False)
def load_table(csv_bytes: bytes):
    return clean(pd.read_csv(BytesIO(csv_bytes)))

with st.sidebar:
    st.markdown('## CloriLife AI')
    st.caption('v0.8 · THUẬN + NGƯỢC + AI TUỔI THỌ · MACHINE LEARNING')
    st.divider()
    st.markdown('**1. Nguồn dữ liệu**')
    source = st.radio('Bộ dữ liệu',
        ['Dryad — 215 mẫu tro bay', 'Tải CSV RCM của tôi'],
        label_visibility='collapsed')
    if source == 'Dryad — 215 mẫu tro bay':
        content = DATASET.read_bytes()
        st.success('Đang dùng dữ liệu thí nghiệm công bố.')
    else:
        file = st.file_uploader('Tải CSV RCM theo mẫu dữ liệu', type=['csv'])
        content = file.getvalue() if file is not None else None
        st.caption('CSV phải có các cột fly_ash_pct, water_binder_ratio, '
                   'cement_type, fly_ash_grade, d_rcm_28_m2_s, source_group.')
    st.markdown('**2. Thuật toán**')
    selected = st.selectbox('Chọn mô hình dự báo', [AUTO] + list(MODELS),
         help='Tự động: chọn mô hình theo MAE GroupKFold nhỏ nhất. Không chọn bằng tập kiểm tra.')
    st.markdown('**3. Nguyên tắc kiểm định**')
    st.caption('Tách ~20% theo nhóm nguồn nghiên cứu làm holdout. '
               'Phần còn lại dùng 5-fold GroupKFold để xếp hạng thuật toán.')
    st.caption('RCM thực nghiệm KHÔNG có nhãn tuổi thọ. Tab mới nhận CSV tuổi thọ có nhãn riêng, không dùng Fick.')
    st.markdown(f'[Nguồn dữ liệu gốc (Dryad)]({SOURCE_URL})')

st.markdown('''
<div class="cl-hero">
  <div class="cl-kicker">RESEARCH PLATFORM · CONCRETE MATERIALS · v0.8</div>
  <div class="cl-title">CloriLife <b>AI</b></div>
  <div class="cl-sub">Dự báo hệ số di chuyển ion clorua bằng <b>9 thuật toán học máy</b>
  trên dữ liệu thí nghiệm bê tông sử dụng tro bay. Hỗ trợ cả <b>bài toán thuận</b>
  và <b>bài toán ngược</b> để tìm cấp phối theo giới hạn RCM mục tiêu;
  <b>không sử dụng định luật Fick</b>. Tab tuổi thọ AI được huấn luyện riêng
  khi có dữ liệu thời gian hư hỏng thực nghiệm có nhãn.</div>
  <div class="cl-pills"><span class="cl-pill">DỮ LIỆU THỰC</span>
  <span class="cl-pill">9 THUẬT TOÁN</span>
  <span class="cl-pill">KIỂM ĐỊNH THEO NGUỒN</span>
  <span class="cl-pill">THIẾT KẾ NGƯỢC CÓ RÀNG BUỘC</span><span class="cl-pill">TUỔI THỌ AI · CẦN NHÃN THỰC</span></div>
</div>
''', unsafe_allow_html=True)

dataset = None
result = None
training_error = None
if not content:
    training_error = 'H\u00e3y t\u1ea3i file CSV RCM h\u1ee3p l\u1ec7 \u0111\u1ec3 b\u1eaft \u0111\u1ea7u hu\u1ea5n luy\u1ec7n.'
else:
    try:
        dataset = load_table(content)
        result = cached_train(content, selected)
    except Exception as e:
        training_error = str(e)
if training_error:
    st.warning(training_error)

if result is None:
    tabs = st.tabs([
        '\U0001f4ca So s\u00e1nh 9 m\u00f4 h\u00ecnh', '\U0001f9ea D\u1ef1 b\u00e1o thu\u1eadn', '\U0001f501 Thi\u1ebft k\u1ebf ng\u01b0\u1ee3c',
        '\U0001f4da D\u1eef li\u1ec7u v\u00e0 ngu\u1ed3n', '\U0001f4cb Ph\u01b0\u01a1ng ph\u00e1p & xu\u1ea5t k\u1ebft qu\u1ea3',
        '⏳ Tuổi thọ AI'])
    for tab in tabs[:-1]:
        with tab:
            st.info('C\u1ea7n b\u1ed9 d\u1eef li\u1ec7u h\u1ee3p l\u1ec7 \u0111\u1ec3 hi\u1ec3n th\u1ecb n\u1ed9i dung tab n\u00e0y.')
    with tabs[-1]:
        render_lifespan_tab(rcm_result=None)
    st.markdown('**D\u1eef li\u1ec7u g\u1ed1c:** '+SOURCE_URL)
else:
    best = result['best_cv']
    champion = result['selected']
    metrics = result['holdout_scores']

    p1, p2, p3, p4 = st.columns(4)
    p1.metric('Số mẫu thực nghiệm', result['total_count'])
    p2.metric('Số nguồn nghiên cứu', dataset['source_group'].nunique())
    p3.metric('Mô hình đang dùng', champion)
    p4.metric('MAE holdout', f"{metrics['MAE_x1e12']:.3f} {UNIT}")

    st.info('**Đầu ra là D_RCM,28 (hệ số di chuyển clorua thử bằng điện trường ở tuổi 28 ngày).** '
            'Đây không phải hệ số khuếch tán biểu kiến D_app và không phải tuổi thọ công trình. '
            'Mô hình chưa được xác nhận để dùng cho hồ sơ thiết kế.')

    model_tab, prediction_tab, inverse_tab, data_tab, method_tab, lifetime_tab = st.tabs([
        '📊 So sánh 9 mô hình', '🧪 Dự báo thuận', '🔁 Thiết kế ngược',
        '📚 Dữ liệu và nguồn', '📋 Phương pháp & xuất kết quả',
        '⏳ Tuổi thọ AI'
    ])

    with model_tab:
        st.subheader('Kết quả kiểm định chéo trên tập huấn luyện')
        st.markdown('<div class="cl-caption">Chỉ số càng nhỏ càng tốt với MAE, RMSE; '
                    'R² càng cao thường càng tốt. Chọn mô hình theo CV, sau đó mới đánh giá holdout.</div>',
                    unsafe_allow_html=True)
        leaderboard = result['table'].copy()
        st.dataframe(leaderboard.style.format({
            'CV MAE (×10⁻¹² m²/s)':'{:.3f}',
            'CV RMSE (×10⁻¹² m²/s)':'{:.3f}',
            'CV R²':'{:.3f}',
        }), use_container_width=True, hide_index=True)
        chart = alt.Chart(leaderboard).mark_bar(cornerRadiusEnd=5).encode(
            x=alt.X('CV MAE (×10⁻¹² m²/s):Q', title='MAE kiểm định chéo (×10⁻¹² m²/s)'),
            y=alt.Y('Thuật toán:N', sort='x', title=None),
            tooltip=['Thuật toán:N',alt.Tooltip('CV MAE (×10⁻¹² m²/s):Q',format='.3f')],
            color=alt.condition(alt.datum['Thuật toán']==best, alt.value('#0f827b'),alt.value('#8ab0be'))
        ).properties(height=350)
        st.altair_chart(chart,use_container_width=True)
        st.markdown('#### Kết quả trên 20% dữ liệu được giữ riêng theo nguồn')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric('Số mẫu kiểm tra',result['test_count'])
        c2.metric('MAE',f"{metrics['MAE_x1e12']:.3f} {UNIT}")
        c3.metric('RMSE',f"{metrics['RMSE_x1e12']:.3f} {UNIT}")
        c4.metric('R²',f"{metrics['R2']:.3f}")
        st.caption(f"Baseline holdout (trung vị D_RCM của train): MAE = {result['baseline_MAE_x1e12']:.3f} {UNIT}. "
                   f"Nguồn train: {result['train_source_count']}, nguồn holdout: {result['test_source_count']}.")
        holdout = result['holdout'].copy()
        lo=float(min(holdout['D_RCM_28_thuc_te_x1e12'].min(), holdout['D_RCM_28_du_bao_x1e12'].min()))
        hi=float(max(holdout['D_RCM_28_thuc_te_x1e12'].max(), holdout['D_RCM_28_du_bao_x1e12'].max()))
        scat=alt.Chart(holdout).mark_circle(size=74,opacity=.76,color='#147c83').encode(
            x=alt.X('D_RCM_28_thuc_te_x1e12:Q',title='Thực nghiệm (×10⁻¹² m²/s)'),
            y=alt.Y('D_RCM_28_du_bao_x1e12:Q',title='Dự báo (×10⁻¹² m²/s)'),
            tooltip=['source_group:N','D_RCM_28_thuc_te_x1e12:Q','D_RCM_28_du_bao_x1e12:Q']
        )
        line=alt.Chart(pd.DataFrame({'x':[lo,hi],'y':[lo,hi]})).mark_line(
            strokeDash=[6,4],color='#999999').encode(x='x:Q',y='y:Q')
        st.altair_chart((scat+line).properties(height=340),use_container_width=True)
        st.download_button('⬇ Tải bảng xếp hạng 9 thuật toán (CSV)',
            leaderboard.to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_9_models_RCM_CV.csv',mime='text/csv')
        st.download_button('⬇ Tải bảng dự báo trên holdout (CSV)',
            holdout.to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_RCM_Holdout.csv',mime='text/csv')

    with prediction_tab:
        st.subheader('Ước lượng hệ số RCM của một cấp phối')
        st.caption('Chỉ thử trong miền dữ liệu huấn luyện. Kết quả là dự báo thống kê, '
                   'không thay thế phép thử tại phòng thí nghiệm.')
        bound_fa = result['bounds']['fly_ash_pct']
        bound_wb = result['bounds']['water_binder_ratio']
        left,right = st.columns(2)
        with left:
            fly_ash = st.number_input('Hàm lượng tro bay (% chất kết dính)',
                min_value=float(bound_fa[0]), max_value=float(bound_fa[1]),
                value=float(min(max(25.0,bound_fa[0]),bound_fa[1])),step=1.0)
            w_b = st.number_input('Tỷ lệ nước/chất kết dính (N/CKD)',
                min_value=float(bound_wb[0]),max_value=float(bound_wb[1]),
                value=float(min(max(0.40,bound_wb[0]),bound_wb[1])), step=0.01, format='%.3f')
        with right:
            cement = st.selectbox('Loại xi măng', result['cement_types'],
                         index=result['cement_types'].index('PO425') if 'PO425' in result['cement_types'] else 0)
            grade = st.selectbox('Cấp tro bay', result['fly_ash_grades'],
                        help='"khong_ghi" là nhãn thiếu dữ liệu gốc, không phải cấp vật liệu mới.')
        if st.button('Dự báo hệ số RCM',type='primary',use_container_width=True):
            try:
                predicted = predict_rcm(result,fly_ash,w_b,cement,grade)
                st.metric('Hệ số D_RCM,28 dự báo',f'{predicted*1e12:.3f} {UNIT}')
                st.success('Dự báo hoàn thành theo mô hình '+champion+'.')
                st.caption('Không quy đổi trực tiếp giá trị này thành thời gian khởi phát ăn mòn hoặc tuổi thọ.')
            except Exception as e:
                st.error(f'Không thể dự báo: {escape(str(e))}')
        st.markdown('#### Kiểm tra phân bố của dữ liệu gốc')
        viz=dataset[['fly_ash_pct','water_binder_ratio','d_rcm_28_m2_s']].copy()
        viz['RCM (×10⁻¹²)']=viz['d_rcm_28_m2_s']*1e12
        viz['Tro bay (%)']=viz['fly_ash_pct']
        st.altair_chart(alt.Chart(viz).mark_circle(size=65,opacity=.68).encode(
            x=alt.X('Tro bay (%):Q',title='Tro bay (% chất kết dính)'),
            y=alt.Y('RCM (×10⁻¹²):Q',title='D_RCM,28 (×10⁻¹² m²/s)'),
            color=alt.Color('water_binder_ratio:Q',title='N/CKD'),
            tooltip=['fly_ash_pct:Q','water_binder_ratio:Q','RCM (×10⁻¹²):Q']
        ).properties(height=330),use_container_width=True)
        st.caption('Biểu đồ điểm cho thấy dữ liệu quan sát, không chứng minh quan hệ nhân quả giữa tro bay và RCM.')


    with inverse_tab:
        st.subheader('Bài toán ngược — Từ RCM mục tiêu đến phương án cấp phối')
        st.markdown('**Dữ liệu vào:** giới hạn tối đa D_RCM,28; khoảng tro bay và N/CKD; '
                    'loại xi măng và cấp tro bay. **Dữ liệu ra:** nhiều phương án cấp phối '
                    'được mô hình dự báo có RCM không vượt mục tiêu.')
        st.info('Đây là **bài toán tìm kiếm có ràng buộc trên mô hình học máy đã huấn luyện**, '
                'không phải phép nghịch đảo duy nhất hay thiết kế cấp phối được kiểm chứng. '
                'Không sử dụng định luật Fick và không dự báo tuổi thọ kết cấu.')
        pairs = training_pairs(result, min_records=3)
        if pairs.empty:
            st.warning('Không đủ dữ liệu huấn luyện cho từng nhóm vật liệu để thiết kế ngược.')
        else:
            descriptions = {
                f"{p.cement_type} | tro bay cấp {p.fly_ash_grade} · {p.n_training_rows} mẫu":
                (str(p.cement_type), str(p.fly_ash_grade))
                for p in pairs.itertuples(index=False)
            }
            chosen_pair = st.selectbox('Chọn nhóm vật liệu có trong tập huấn luyện',
                                       list(descriptions), key='inv_material_group')
            cmt, grd = descriptions[chosen_pair]
            group = result['training_rows'].loc[
                (result['training_rows']['cement_type'] == cmt) &
                (result['training_rows']['fly_ash_grade'] == grd)]
            fa_min, fa_max = float(group['fly_ash_pct'].min()),float(group['fly_ash_pct'].max())
            wb_min, wb_max = float(group['water_binder_ratio'].min()),float(group['water_binder_ratio'].max())
            st.caption(f'Miền quan sát của nhóm: tro bay {fa_min:g}–{fa_max:g}%; '
                       f'N/CKD {wb_min:.3f}–{wb_max:.3f}. '
                       'Không ghép tùy tiện các loại xi măng và cấp tro bay chưa có trong dữ liệu.')

            with st.form('inverse_input'):
                left, right = st.columns(2)
                with left:
                    desired = st.number_input('RCM mục tiêu tối đa (×10⁻¹² m²/s)',
                        min_value=0.1,max_value=40.0,value=8.0,step=0.5,format='%.2f')
                    operation = st.selectbox('Cần tìm đại lượng nào?',MODES)
                    preference = st.selectbox('Ưu tiên khi xếp hạng phương án', PREFERENCES)
                    include_new = st.checkbox('Cho phép phương án nội suy gần mẫu đã đo',value=True,
                        help='Chỉ nội suy trong miền lồi của cùng nhóm vật liệu, gần cấp phối thực nghiệm. '
                             'Phương án nội suy chưa được thí nghiệm.')
                with right:
                    fa_interval = st.slider('Khoảng hàm lượng tro bay (%)',
                        min_value=fa_min,max_value=fa_max,
                        value=(fa_min,fa_max),step=0.5) if fa_max > fa_min else (fa_min,fa_max)
                    wb_interval = st.slider('Khoảng N/CKD',
                        min_value=wb_min,max_value=wb_max,
                        value=(wb_min,wb_max),step=0.005,format='%.3f') if wb_max > wb_min else (wb_min,wb_max)
                    locked_wb = st.number_input('N/CKD cố định (khi chỉ tìm tro bay)',
                        min_value=wb_min,max_value=wb_max,
                        value=float(np.median(group['water_binder_ratio'])),step=0.005,format='%.3f',
                        disabled=(operation != MODES[1]))
                    locked_fa = st.number_input('Tro bay cố định (khi chỉ tìm N/CKD)',
                        min_value=fa_min,max_value=fa_max,
                        value=float(np.median(group['fly_ash_pct'])),step=0.5,
                        disabled=(operation != MODES[2]))
                    safety_buffer = st.selectbox('Biên sàng lọc thêm so với mục tiêu',
                        [0.0,0.5,1.0],format_func=lambda x:
                        ('Không thêm' if x==0 else f'Trừ {x:g} × MAE kiểm định chéo'),
                        help='Chỉ là biên dự phòng mang tính kinh nghiệm, không phải khoảng tin cậy.')
                submitted = st.form_submit_button('🔎 TÌM CÁC PHƯƠNG ÁN CẤP PHỐI',
                                                   type='primary',use_container_width=True)
            current_sig = (hashlib.sha256(content).hexdigest(), champion, cmt, grd)
            if submitted:
                try:
                    solutions = inverse_search(
                        result, target_rcm_x1e12=desired, cement_type=cmt, fly_ash_grade=grd,
                        fly_ash_range=fa_interval, wb_range=wb_interval,
                        mode=operation, fixed_wb=locked_wb if operation==MODES[1] else None,
                        fixed_fly_ash=locked_fa if operation==MODES[2] else None,
                        preference=preference, include_interpolations=include_new,
                        buffer_mae_multiple=safety_buffer)
                    st.session_state['inverse_run'] = {'sig':current_sig,'result':solutions,
                         'model':champion, 'cement':cmt, 'grade':grd}
                except (ValueError, KeyError) as e:
                    st.session_state.pop('inverse_run',None)
                    st.error(f'Không tính được bài toán ngược: {escape(str(e))}')

            saved=st.session_state.get('inverse_run')
            if saved and saved['sig'] == current_sig:
                solutions=saved['result']
                st.divider()
                a,b,c,d=st.columns(4)
                a.metric('RCM mục tiêu',f"{solutions['target_rcm_x1e12']:.2f} {UNIT}")
                gate=solutions['screening_limit_x1e12']
                b.metric('Ngưỡng sàng lọc',f'{gate:.2f} {UNIT}' if np.isfinite(gate) else '—')
                c.metric('Phương án đã xét',solutions['candidate_count'])
                d.metric('Ph\u01b0\u01a1ng \u00e1n s\u00e0ng l\u1ecdc',solutions['feasible_count'])
                st.caption(f"M\u1eabu \u0111o th\u1ef1c t\u1ebf \u0111\u1ea1t: {solutions['measured_pass_count']}; "
                           f"n\u1ed9i suy ch\u1ec9 d\u1ef1 b\u00e1o \u0111\u1ea1t: {solutions['predicted_only_count']}; "
                           f"m\u1eabu \u0111o d\u1ef1 b\u00e1o \u0111\u1ea1t nh\u01b0ng \u0111o kh\u00f4ng \u0111\u1ea1t: {solutions['false_positive_count']}")
                holdout_check = result['holdout']
                holdout_pred_pass = holdout_check['D_RCM_28_du_bao_x1e12'].le(desired)
                holdout_real_fail = holdout_check['D_RCM_28_thuc_te_x1e12'].gt(desired)
                n_pred_pass = int(holdout_pred_pass.sum())
                n_false_pass = int((holdout_pred_pass & holdout_real_fail).sum())
                st.warning(f'Ki\u1ec3m tra c\u1ea3nh b\u00e1o tr\u00ean holdout (ch\u1ec9 \u0111\u1ec3 \u0111\u00e1nh gi\u00e1, kh\u00f4ng ch\u1ecdn ph\u01b0\u01a1ng \u00e1n): '
                           f'{n_false_pass}/{n_pred_pass} m\u1eabu d\u1ef1 b\u00e1o \u0111\u1ea1t nh\u01b0ng th\u1ef1c \u0111o kh\u00f4ng \u0111\u1ea1t '
                           f'khi ng\u01b0\u1ee1ng l\u00e0 {desired:g} {UNIT}. '
                           'M\u1eabu th\u1eed thu\u1ed9c c\u00e1c ngu\u1ed3n gi\u1eef ri\u00eang. ')
                if solutions['status']=='candidates_found':
                    st.success('C\u00f3 m\u1eabu \u0111o \u0111\u1ea1t ng\u01b0\u1ee1ng v\u00e0/ho\u1eb7c c\u1ea5p ph\u1ed1i n\u1ed9i suy d\u1ef1 b\u00e1o \u0111\u1ea1t. '
                               'C\u00e1c ph\u01b0\u01a1ng \u00e1n n\u1ed9i suy ch\u01b0a c\u00f3 k\u1ebft qu\u1ea3 \u0111o th\u1ef1c t\u1ebf.')
                elif solutions['status']=='no_candidates_meet_limit':
                    st.warning('Không có phương án nào đạt mục tiêu trong các ràng buộc hiện tại. '
                               '**Bảng sau chỉ là các phương án gần đạt, KHÔNG đạt yêu cầu.** '
                               'Hãy nới khoảng đầu vào hoặc dùng vật liệu khác có dữ liệu.')
                else:
                    st.warning(solutions['notes'])
                short=solutions['shortlisted']
                if not short.empty:
                    view=short[['origin','fly_ash_pct','water_binder_ratio',
                                'predicted_rcm_x1e12','observed_rcm_x1e12',
                                'support_distance','source_excel_row','pass_predicted_limit',
                            'pass_measured_limit','evidence_level','prediction_vs_measurement_conflict']].copy()
                    view.columns=['Lo\u1ea1i ph\u01b0\u01a1ng \u00e1n','Tro bay (%)','N/CKD',
                                  'RCM d\u1ef1 b\u00e1o (\u00d710\u207b\u00b9\u00b2)','RCM \u0111o (\u00d710\u207b\u00b9\u00b2)',
                                  'Kho\u1ea3ng c\u00e1ch \u0111\u1ebfn m\u1eabu','D\u00f2ng Excel g\u1ed1c',
                                  'D\u1ef1 b\u00e1o \u0111\u1ea1t','M\u1eabu \u0111o \u0111\u1ea1t',
                                  'C\u01a1 s\u1edf k\u1ebft lu\u1eadn','D\u1ef1 b\u00e1o v\u00e0 \u0111o m\u00e2u thu\u1eabn']
                    st.dataframe(view.style.format({'Tro bay (%)':'{:.2f}','N/CKD':'{:.3f}',
                        'RCM dự báo (×10⁻¹²)':'{:.3f}','RCM đo (×10⁻¹²)':'{:.3f}',
                        'Khoảng cách đến mẫu':'{:.2f}'}),use_container_width=True,hide_index=True)
                    fig=short.copy()
                    fig['Dự báo RCM (×10⁻¹²)']=fig['predicted_rcm_x1e12']
                    fig['Tro bay (%)']=fig['fly_ash_pct']
                    fig['N/CKD']=fig['water_binder_ratio']
                    plot=alt.Chart(fig).mark_circle(size=150,opacity=.82).encode(
                        x=alt.X('Tro bay (%):Q',title='Tro bay (% chất kết dính)'),
                        y=alt.Y('N/CKD:Q',title='Tỷ lệ N/CKD'),
                        color=alt.Color('Dự báo RCM (×10⁻¹²):Q',title='RCM dự báo'),
                        shape=alt.Shape('origin:N',title='Nguồn phương án'),
                        tooltip=['origin:N','Tro bay (%):Q','N/CKD:Q',
                                 alt.Tooltip('Dự báo RCM (×10⁻¹²):Q',format='.3f')]
                    ).properties(height=340)
                    st.altair_chart(plot,use_container_width=True)
                    csv=short.to_csv(index=False).encode('utf-8-sig')
                    st.download_button('⬇ Tải các phương án và nguồn tham khảo (CSV)',csv,
                        file_name='CloriLife_BaiToanNguoc_RCM.csv',mime='text/csv')
                    st.caption('“Mẫu thực nghiệm”: đầu vào đã được đo; cột RCM đo lấy từ dữ liệu gốc. '
                               '“Nội suy”: cấp phối được đề xuất bằng mô hình, RCM đo để trống. '
                               'Khoảng cách hỗ trợ là khoảng cách không thứ nguyên đến mẫu huấn luyện gần nhất; '
                               'không phải xác suất tin cậy. Đầu ra có thể không duy nhất.')
                conflicts = solutions.get('all_candidates', pd.DataFrame())
                if not conflicts.empty:
                    conflicts = conflicts.loc[conflicts['prediction_vs_measurement_conflict']]
                    if not conflicts.empty:
                        with st.expander('\u26a0\ufe0f M\u1eabu \u0111o m\u00e2u thu\u1eabn v\u1edbi d\u1ef1 b\u00e1o', expanded=False):
                            st.warning('\u0110\u00e2y l\u00e0 c\u00e1c m\u1eabu th\u1ef1c t\u1ebf \u0111o kh\u00f4ng kh\u1edbp v\u1edbi ph\u00e2n lo\u1ea1i d\u1ef1 b\u00e1o. '
                                       'Kh\u00f4ng \u0111\u01b0\u1ee3c xem d\u1ef1 b\u00e1o l\u00e0 b\u1eb1ng ch\u1ee9ng th\u1eed nghi\u1ec7m.')
                            st.dataframe(conflicts[['fly_ash_pct','water_binder_ratio',
                                'observed_rcm_x1e12','predicted_rcm_x1e12','source_excel_row',
                                'evidence_level']],use_container_width=True,hide_index=True)
                st.error('**Không dùng ngay cho thiết kế thi công.** Cần thí nghiệm RCM xác nhận và '
                         'kiểm tra riêng cường độ nén, tính công tác, độ bền lâu, đặc tính tro bay '
                         'và các yêu cầu kỹ thuật khác. Dữ liệu hiện tại thiếu nhiều thông số đó.')
                st.caption(f'Mô hình dùng để tìm nghiệm: {saved["model"]}. '
                           f'Nhóm vật liệu: {saved["cement"]}, cấp tro bay {saved["grade"]}. '
                           'Tìm nghiệm từ tập huấn luyện, không dùng 20% holdout để đề xuất.')

    with data_tab:
        st.subheader('Nguồn dữ liệu và khả năng truy xuất')
        st.write('**Ngu\u1ed3n:** LuFeng Yang, En Zhu, YuChen Wei (2025); Dryad DOI: 10.5061/dryad.66t1g1k6j.')
        st.link_button('Mở dữ liệu gốc và tài liệu phương pháp', SOURCE_URL)
        st.warning('L\u01b0u \u00fd tr\u00edch ngu\u1ed3n: Excel phi\u00ean b\u1ea3n t\u1ea3i v\u1ec1 c\u00f3 215 d\u00f2ng Table 5; '
                   'l\u1ecbch s\u1eed c\u1eadp nh\u1eadt Dryad n\u00eau 209 m\u1eabu v\u00e0 lo\u1ea1i 13 b\u1ea3n ghi tr\u00f9ng. '
                   'C\u1ea7n \u0111\u1ed1i chi\u1ebfu ch\u00eanh l\u1ec7ch tr\u01b0\u1edbc khi n\u1ed9p b\u00e1o c\u00e1o.')

        st.write('Số mẫu sau kiểm tra:',len(dataset), '— số nhóm nguồn:',dataset['source_group'].nunique())
        cols = ['fly_ash_pct','water_binder_ratio','cement_type','fly_ash_grade',
                'd_rcm_28_m2_s','source_reference','source_excel_row']
        show = dataset[[v for v in cols if v in dataset.columns]].copy()
        if 'd_rcm_28_m2_s' in show:
            show['D_RCM (×10⁻¹² m²/s)']=show.pop('d_rcm_28_m2_s')*1e12
        st.dataframe(show,hide_index=True,use_container_width=True,height=380)
        st.download_button('⬇ Tải bộ dữ liệu gốc đã chuẩn hóa (CSV)',
            content, file_name='CloriLife_RCM_Real_FlyAsh_215.csv',mime='text/csv')
        st.warning('Dữ liệu gồm kết quả đo RCM. Không gọi đây là dữ liệu tuổi thọ; '
                   'không tự điền cường độ, nhiệt độ hoặc tuổi thọ nếu bảng gốc thiếu.')
        st.caption('Các giá trị cấp tro bay còn trống trong tệp gốc được bảo toàn; '
                   'pipeline học máy chỉ dùng nhãn thiếu nội bộ khi mã hóa biến phân loại.')

    with method_tab:
        st.subheader('Tại sao sử dụng học máy cho cả bài toán thuận và ngược?')
        st.write('Đề tài phiên bản này nghiên cứu **mối liên hệ giữa thông số cấp phối và hệ số di chuyển ion clorua RCM**, '
                 'thay vì tính nồng độ theo chiều sâu hoặc thời gian khởi phát ăn mòn.')
        st.markdown('''
    **Quy trình nghiên cứu**

    1. Thu thập dữ liệu thí nghiệm RCM có dẫn nguồn (ưu tiên mẫu bê tông tro bay).
    2. Làm sạch dữ liệu, giữ nguyên phương pháp thử và đơn vị mục tiêu.
    3. Tách dữ liệu theo **nguồn nghiên cứu**, tránh mẫu cùng nguồn xuất hiện ở cả train và test.
    4. Đánh giá 9 thuật toán bằng **GroupKFold** trên dữ liệu huấn luyện.
    5. Chọn thuật toán có MAE CV nhỏ nhất; đánh giá lần cuối trên **holdout riêng biệt**.
    6. **Bài toán thuận:** nhập cấp phối để dự báo D_RCM,28.
    7. **Bài toán ngược:** nhập ngưỡng D_RCM,28; tìm nhiều tổ hợp tro bay và N/CKD khả thi theo mô hình, ở trong miền lồi và gần các mẫu thực nghiệm cùng loại vật liệu.
    8. Thí nghiệm xác nhận mọi phương án được đề xuất; không dùng kết quả mô hình thay thế thiết kế cấp phối.

    **Giới hạn khoa học:** Dữ liệu RCM không trực tiếp cho nhãn tuổi thọ. Tab mới chỉ
    dự báo thời gian đến trạng thái giới hạn **khi tải riêng dữ liệu thời gian quan sát có nhãn**.
    Không thay thế đánh giá năng lực chịu lực và kiểm định công trình.
    ''')
        st.markdown('**Các thuật toán đang hỗ trợ:** '+', '.join(MODELS)+'.')
        st.markdown('**Thông số đầu vào thực sự dùng:** Tro bay (%), N/CKD, loại xi măng, cấp tro bay.')
        st.markdown('**Biến đích:** D_RCM,28 (m²/s), theo phương pháp Rapid Chloride Migration.')
        st.caption('Một số mô hình được cấu hình sẵn phục vụ đối chứng; tập holdout không được sử dụng để chọn tham số hay chọn thuật toán.')
        report = (
            'CLORILIFE AI v0.8 - BAO CAO TOM TAT\n'
            'Phuong phap: Machine Learning, KHONG SU DUNG FICK\n'
            f'Nguon du lieu: {SOURCE_URL}\n'
            f'So mau: {result["total_count"]}; so nhom nguon: {dataset["source_group"].nunique()}\n'
            f'Mo hinh chon: {champion}; mo hinh CV tot nhat: {best}\n'
            f'So mau train: {result["train_count"]}; holdout: {result["test_count"]}\n'
            f'Holdout MAE x1e12: {metrics["MAE_x1e12"]:.6f}\n'
            f'Holdout RMSE x1e12: {metrics["RMSE_x1e12"]:.6f}\n'
            f'Holdout R2: {metrics["R2"]:.6f}\n'
            'Luu y: D_RCM KHONG PHAI thoi gian khoi phat an mon hay tuoi tho ket cau.\n'
        )
        st.download_button('⬇ Tải bản tóm tắt kết quả (TXT)',
            report.encode('utf-8-sig'), file_name='CloriLife_ML_only_tomtat.txt',mime='text/plain')

    with lifetime_tab:
        render_lifespan_tab(rcm_result=result)

    st.markdown('<div class="cl-foot">CloriLife AI v0.8 · Thuận + ngược · Research demo · Dữ liệu RCM được công bố · '
                'Tuổi thọ AI chỉ khi có nhãn thời gian quan sát thực; không dùng quyết định thiết kế</div>',unsafe_allow_html=True)
