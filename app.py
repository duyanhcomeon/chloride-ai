"""CloriLife AI v0.4.1: chloride ingress + conditional uniform-corrosion scenarios.

Extends the v0.3 prototype with a two-phase steel area-loss threshold
scenario. This is not a structural capacity or code compliance analysis.
"""
from __future__ import annotations

from html import escape
import base64
from io import BytesIO
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from comparison import DEFAULT_SCENARIOS, run_comparison
from corrosion import (serviceability_indicator, remaining_bar_history,
                       current_density_scenarios, area_loss_percent,
                       COL_TIME, COL_REMAIN, COL_ICORR, COL_T_CRACK, COL_T_PROP, COL_T_LIMIT)
from ml import MODEL_NAMES, DEFAULT_MODEL, REQUIRED, fit_model, predict, repeated_cv

SHORT_NAMES = {'ridge': 'Ridge', 'rf': 'Random Forest', 'hgb': 'Gradient Boosting',
               'svr': 'SVR', 'gpr': 'Gaussian Process'}
from pdf_report import build_pdf_report
from physics import concentration, initiation_years

st.set_page_config(
    page_title='CloriLife AI v0.4.1 | Concrete Durability Intelligence',
    page_icon='🧬', layout='wide', initial_sidebar_state='expanded',
)

# HTML below contains fixed app-controlled markup; user-provided text is escaped.
st.markdown(
    '<style>\n    /* Layout and typography */\n    [data-testid="stAppViewContainer"] {background: #f5f8fc;}\n    .main .block-container {max-width: 1420px; padding-top: 1.25rem; padding-bottom: 2.8rem;}\n    [data-testid="stSidebar"] {background: #edf3f8; border-right: 1px solid #dbe4ed;}\n    [data-testid="stSidebar"] .block-container {padding-top: 1.3rem;}\n    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #183b55;}\n    [data-testid="stMetricValue"] {font-weight: 750; color: #173a54;}\n    div[data-testid="stAlert"] {border-radius: 12px;}\n    [data-testid="stForm"] {border-radius: 14px;}\n    .stTabs [data-baseweb="tab-list"] {gap: 18px;}\n    .stTabs [data-baseweb="tab"] {font-weight: 700;}\n    button[kind="primary"] {background-color: #137e91; border-color: #137e91;}\n\n    /* Brand hero */\n    .cl-hero {\n        position: relative; overflow: hidden; border-radius: 19px;\n        padding: 30px 36px 32px 36px;\n        background: radial-gradient(circle at 89% 18%, rgba(79,213,206,.21), transparent 28%),\n                    linear-gradient(108deg, #102741 0%, #173e5d 63%, #126c79 100%);\n        box-shadow: 0 12px 28px rgba(25,55,81,.12);\n        color: white; margin-bottom: 16px;\n    }\n    .cl-hero:after {\n        content: ""; position: absolute; height: 230px; width: 230px;\n        border-radius: 50%; border: 1px solid rgba(176,245,235,.20);\n        right: -56px; top: -100px; box-shadow: 0 0 0 40px rgba(176,245,235,.04), 0 0 0 84px rgba(176,245,235,.03);\n    }\n    .cl-eyebrow {letter-spacing: 2.2px; color: #9ee1da; font-size: 11px; font-weight: 800;}\n    .cl-title {font-size: clamp(34px, 3.5vw, 57px); line-height: 1.05; font-weight: 850;\n        letter-spacing: -1.5px; color: #fff; margin: 13px 0 12px 0;}\n    .cl-title span {color: #70e3d2;}\n    .cl-description {max-width: 900px; font-size: 15px; line-height: 1.65;\n        color: #e4edf4; margin-bottom: 19px;}\n    .cl-pill {border: 1px solid rgba(210,244,239,.35); background: rgba(240,252,250,.09);\n        border-radius: 30px; padding: 6px 12px; font-size: 11px; color: #def4f2;\n        display: inline-block; margin-right: 7px; margin-bottom: 3px; font-weight: 700;}\n    .cl-sidebar-brand {padding: 12px 13px 13px 13px; background: #163b59; color: #fff;\n        border-radius: 13px; margin-bottom: 19px;}\n    .cl-sidebar-brand strong {font-size: 19px; letter-spacing: -.3px;}\n    .cl-sidebar-brand strong span {color: #7fe3d6;}\n    .cl-sidebar-brand small {display: block; color: #b9d2df; margin-top: 4px; font-size: 11px;}\n    .cl-section {font-size: 12px; letter-spacing: 1.3px; text-transform: uppercase;\n        color: #337187; font-weight: 850; margin: 12px 0 8px 0;}\n    .cl-kpi {background: #fff; border: 1px solid #e2eaf0; border-radius: 15px;\n        padding: 18px 19px; min-height: 119px; box-shadow: 0 3px 12px rgba(17,52,75,.035);}\n    .cl-kpi-label {font-size: 12px; color: #617a8a; font-weight: 700; line-height: 1.4;}\n    .cl-kpi-value {font-size: clamp(24px, 2.3vw, 37px); color: #133b58;\n        font-weight: 850; line-height: 1.3; letter-spacing: -0.8px; margin: 4px 0;}\n    .cl-kpi-value.teal {color: #047b80;}\n    .cl-kpi-sub {font-size: 11px; color: #728d9a;}\n    .cl-note {border-left: 4px solid #d59b45; padding: 13px 17px; background: #fff8e9;\n        border-radius: 0 11px 11px 0; font-size: 13px; line-height: 1.65;\n        color: #65502c; margin: 12px 0 20px 0;}\n    .cl-panel-title {font-size: 20px; color: #173a54; font-weight: 850;\n        margin: 12px 0 3px 0;}\n    .cl-panel-sub {color: #5e788a; font-size: 13px; margin-bottom: 16px;}\n    .cl-info-panel {background: white; border: 1px solid #e3edf3;\n        border-radius: 14px; padding: 18px; font-size: 13px; line-height: 1.72; color: #3b5667;}\n    .cl-info-panel strong {color: #173b57;}\n    .cl-foot {color: #8293a0; font-size: 11px; text-align: center;\n        padding: 21px 12px 2px;}\n\n    .cl-small-chip {display:inline-block;background:#e8f3f4;color:#166773;border-radius:999px;padding:5px 10px;font-size:11px;font-weight:700;margin-right:6px;}\n    .cl-outline {border:1px solid #dfebf2;border-radius:12px;padding:14px 16px;background:white;margin-bottom:15px;}\n    .cl-section-padding {margin-top:16px;}\n    .cl-tab-intro {font-size:13px;color:#5a7888;margin-bottom:14px;}\n</style>',
    unsafe_allow_html=True,
)


# A subtle, compressed concrete surface, bundled with the repository.
# Encoded once per server process (cached) and embedded as a data URI, so no
# third-party image server or static-file configuration is needed.
@st.cache_resource(show_spinner=False)
def texture_css() -> str:
    path = Path(__file__).with_name('concrete_texture.jpg')
    if not path.exists():
        return ''
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return (
        "<style>[data-testid='stAppViewContainer']{"
        "background:linear-gradient(rgba(246,249,252,.74),rgba(246,249,252,.80)),"
        f"url('data:image/jpeg;base64,{encoded}') center top/cover fixed!important;}}"
        ".cl-hero{background:linear-gradient(108deg,rgba(16,39,65,.91),"
        f"rgba(18,92,107,.88)),url('data:image/jpeg;base64,{encoded}') center/cover!important;}}"
        "</style>")


_texture_css = texture_css()
if _texture_css:
    st.markdown(_texture_css, unsafe_allow_html=True)


def info_card(label, value, note, emphasized=False):
    cls = 'teal' if emphasized else ''
    return (
        '<div class="cl-kpi">'
        f'<div class="cl-kpi-label">{escape(str(label))}</div>'
        f'<div class="cl-kpi-value {cls}">{escape(str(value))}</div>'
        f'<div class="cl-kpi-sub">{escape(str(note))}</div>'
        '</div>'
    )


@st.cache_resource(show_spinner=False)
def train_from_csv(raw_bytes: bytes, model_name: str):
    """Cache fitting per (dataset, model); the holdout is never used for fitting."""
    df = pd.read_csv(BytesIO(raw_bytes))
    return fit_model(df, model_name)


# SIDEBAR - shared input state for all tabs.
with st.sidebar:
    st.markdown(
        '<div class="cl-sidebar-brand"><strong>CloriLife <span>AI</span></strong>'
        '<small>CONCRETE DURABILITY LAB · v0.4.1</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown('**01 — Dữ liệu huấn luyện AI (RCM)**')
    source_choice = st.selectbox('Nguồn dữ liệu AI', [
        'Chưa sử dụng bộ dữ liệu',
        'Bộ dữ liệu RCM từ tài liệu',
        'CSV thí nghiệm của tôi',
    ], help='AI dự báo hệ số RCM (di trú nhanh), KHÔNG phải D28. Kết quả AI không đưa vào mô hình khuếch tán chính.')
    training_bytes = None
    data_label = 'Chưa có'
    if source_choice == 'Bộ dữ liệu RCM từ tài liệu':
        try:
            training_bytes = Path(__file__).with_name('chloride_RCM_tidy.csv').read_bytes()
            data_label = 'Bộ dữ liệu RCM từ tài liệu'
        except OSError:
            st.error('Không tìm thấy chloride_RCM_tidy.csv cùng thư mục với app.py. '
                     'Hãy tải file này lên GitHub hoặc chọn nguồn dữ liệu khác.')
    elif source_choice == 'CSV thí nghiệm của tôi':
        upload = st.file_uploader('Tải CSV thí nghiệm', type=['csv'], key='training_csv')
        if upload is not None:
            training_bytes = upload.getvalue()
            data_label = 'CSV do người dùng tải lên'
        st.caption('Cột bắt buộc: ' + ', '.join(REQUIRED) + '. Đơn vị D_RCM là ×10⁻¹² m²/s.')
    data_label_en = {'Bộ dữ liệu RCM từ tài liệu': 'RCM literature dataset',
                     'CSV do người dùng tải lên': 'user-uploaded CSV'}.get(data_label, 'none')
    st.caption('Chọn và huấn luyện mô hình AI ở tab “Đánh giá AI”.')
    model, metrics, bounds = None, None, None
    st.divider()
    st.markdown('**02 — Bê tông tro bay và cấp phối**')
    fly_ash = st.number_input('Tro bay / chất kết dính (%)', 0.0, 80.0, 25.0, 1.0)
    slag = st.number_input('Xỉ / chất kết dính (%)', 0.0, 80.0, 0.0, 1.0)
    wb = st.number_input('Nước / chất kết dính', 0.15, 0.90, 0.40, 0.01)
    cement_family = st.selectbox('Họ xi măng', ['PO', 'PII', 'PI'])
    cement_grade = st.selectbox('Cấp xi măng', ['32.5', '42.5', '42.5R', '52.5', '52.5R'], index=1)
    method = st.selectbox('Phương pháp thử RCM (cho AI)', ['not_stated', 'constant', 'variable'],
                          format_func=lambda k: {'not_stated': 'Không rõ / không ghi',
                                                 'constant': 'Điện áp không đổi',
                                                 'variable': 'Điện áp thay đổi'}[k])
    if fly_ash + slag > 100:
        st.error('Tro bay + xỉ không được vượt 100% chất kết dính.')
    ai_base = {
        'water_binder_ratio': wb,
        'fly_ash_fraction_of_binder': fly_ash / 100,
        'slag_fraction_of_binder': slag / 100,
        'cement_family': cement_family,
        'cement_grade': cement_grade,
        'method': method,
    }
    st.divider()
    st.markdown('**D28 cho bài toán khuếch tán chính**')
    d28 = st.number_input(
        'D28 đo được / tham khảo (×10⁻¹² m²/s)',
        min_value=0.01, max_value=100.0, value=8.0, step=0.5
    ) * 1e-12
    st.caption('D28 là hệ số khuếch tán tự nhiên sau 28 ngày. Hệ số RCM của bộ dữ liệu KHÔNG được dùng thay thế D28.')
    m = st.number_input('Số mũ già hóa m (giả định / hiệu chuẩn)', 0.0, 0.95, 0.35, 0.05)
    st.caption('Dữ liệu mặc định chỉ để minh họa. Không sử dụng để thiết kế kết cấu.')
    st.divider()
    st.markdown('**03 — Kịch bản ăn mòn phát triển**')
    bar_diameter = st.number_input(
        'Đường kính cốt thép ban đầu (mm)',
        min_value=6.0, max_value=50.0, value=16.0, step=2.0,
        help='Dùng đường kính thanh thép giả định hoặc thông số đo thực tế.')
    i_corr = st.number_input(
        'Mật độ dòng ăn mòn i_corr (µA/cm²)',
        min_value=0.01, max_value=20.0, value=1.0, step=0.25,
        help='Phải đo/hiệu chuẩn hoặc đặt theo kịch bản. AI hiện chưa dự báo i_corr.')
    area_limit = st.slider(
        'Ngưỡng hao hụt tiết diện cốt thép (%)',
        min_value=1, max_value=40, value=10, step=1,
        help='Giới hạn người dùng tự đặt để minh họa, KHÔNG là ngưỡng phá hoại kết cấu.')
    crack_um = st.number_input(
        'Độ xuyên sâu gây nứt lớp bảo vệ (µm)',
        min_value=5.0, max_value=500.0, value=50.0, step=5.0,
        help='Giá trị GIẢ ĐỊNH để minh họa mốc nứt. Phụ thuộc lớp bảo vệ, đường kính thép và '
             'chất lượng bê tông; cần hiệu chuẩn bằng số liệu thí nghiệm/hiện trường.')
    st.caption('Mô hình ăn mòn ĐỀU, dòng ăn mòn không đổi. Chưa xét rỗ cục bộ, nứt, bám dính, tải trọng hay suy giảm khả năng chịu lực.')



st.markdown('''
<div class="cl-hero">
  <div class="cl-eyebrow">APPLIED MACHINE LEARNING  /  DURABILITY ENGINEERING</div>
  <div class="cl-title">CloriLife <span>AI</span></div>
  <div class="cl-description">Nền tảng phân tích xâm nhập ion clorua và thời gian đạt ngưỡng ăn mòn
  cốt thép trong bê tông sử dụng tro bay. So sánh cấp phối, kiểm định mô hình học máy
  và đánh giá sơ bộ giai đoạn ăn mòn phát triển đến một ngưỡng hao hụt thép tùy chọn.</div>
  <span class="cl-pill">FLY ASH MIX COMPARISON</span>
  <span class="cl-pill">FICK DIFFUSION</span>
  <span class="cl-pill">AI VALIDATION</span>
  <span class="cl-pill">PDF REPORT</span>
  <span class="cl-pill">UNIFORM CORROSION SCENARIOS</span>
  <span class="cl-pill">RESEARCH MVP · v0.4.1</span>
</div>''', unsafe_allow_html=True)
st.markdown(
    '<div class="cl-note"><b>Giới hạn áp dụng:</b> Phần mềm tính thời điểm Cl⁻ tại cốt thép '
    'đạt ngưỡng và tiếp tục ước tính thời gian ăn mòn ĐỀU đến một tỷ lệ hao hụt thép '
    'do người dùng lựa chọn. <b>Đây KHÔNG phải tuổi thọ chịu lực được kiểm định.</b> '
    'Chưa xét ăn mòn rỗ, nứt, suy giảm độ bám dính, tải trọng và an toàn kết cấu.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="cl-section">03 / KẾT CẤU VÀ MÔI TRƯỜNG</div>', unsafe_allow_html=True)
a,b,c,d = st.columns(4, gap='medium')
with a:
    cover = st.number_input('Lớp bê tông bảo vệ (mm)',5.0,200.0,50.0,1.0)
with b:
    cs = st.number_input('Clorua bề mặt Cs (% khối lượng CKD)',0.01,10.0,1.20,0.10)
with c:
    c0 = st.number_input('Clorua ban đầu C0 (% khối lượng CKD)',0.00,10.0,0.00,0.05)
with d:
    critical = st.number_input('Ngưỡng tới hạn Ccrit (% khối lượng CKD)',0.01,10.0,0.40,0.05)
x,y = st.columns([1,1])
with x:
    horizon = st.slider('Giới hạn phân tích (năm)',10,200,100,5)
with y:
    st.caption('Cs, C0, Ccrit phải tính theo cùng cơ sở khối lượng chất kết dính. Ngưỡng tới hạn là đầu vào giả định, cần cơ sở thí nghiệm/khoa học.')

ready, blocked_kind, blocked_msg = True, 'error', ''
years, time_axis, chloride_axis, c_10, service = None, None, None, None, None
if not (c0 < critical < cs):
    ready, blocked_msg = False, 'Dữ liệu không hợp lệ: cần C0 < Ccrit < Cs.'
else:
    try:
        years = initiation_years(cover,d28,m,cs,critical,c0,horizon)
        time_axis = np.linspace(0.001,horizon,360)
        chloride_axis = np.asarray(concentration(time_axis,cover,d28,m,cs,c0),dtype=float)
        c_10 = float(concentration(min(10,horizon),cover,d28,m,cs,c0))
        service = serviceability_indicator(years,bar_diameter,i_corr,area_limit,crack_um)
    except ValueError as error:
        ready, blocked_msg = False, f'Tham số vật lý không hợp lệ: {error}'

if ready:
    st.markdown('<div class="cl-section">04 / TÓM TẮT KẾT QUẢ</div>', unsafe_allow_html=True)
    p1,p2,p3=st.columns([1.28,1,1],gap='medium')
    with p1:
        st.markdown(info_card('THỜI GIAN ĐẠT NGƯỠNG CL⁻',
                    f'{years:.1f} năm' if years is not None else f'> {horizon} năm',
                    'Mốc khởi phát ăn mòn giả định' if years is not None else 'Chưa đạt trong thời gian phân tích',True),
                    unsafe_allow_html=True)
    with p2:
        st.markdown(info_card('HỆ SỐ KHUẾCH TÁN D28',f'{d28/1e-12:.2f} ×10⁻¹²','m²/s · tuổi tham chiếu 28 ngày'),unsafe_allow_html=True)
    with p3:
        st.markdown(info_card('CL⁻ TẠI THÉP NĂM THỨ 10',f'{c_10:.3f}%','Cùng cơ sở khối lượng chất kết dính'),unsafe_allow_html=True)
    st.markdown('<div class="cl-section">05 / CHỈ SỐ TUỔI THỌ SƠ BỘ THEO MỐC NỨT VÀ HAO HỤT THÉP</div>',unsafe_allow_html=True)
    s1,s2,s3,s4=st.columns(4, gap='medium')
    unknown_txt = 'Chưa xác định'
    with s1:
        st.markdown(info_card('GIAI ĐOẠN KHỞI PHÁT',
                f'{years:.1f} năm' if years is not None else f'> {horizon} năm',
                'Mô hình xâm nhập ion clorua'),unsafe_allow_html=True)
    with s2:
        st.markdown(info_card('MỐC NỨT LỚP BẢO VỆ (GIẢ ĐỊNH)',
                f"{service['cracking_total_years']:.1f} năm" if service['cracking_total_years'] is not None else unknown_txt,
                f"+{service['cracking_years']:.1f} năm sau khởi phát · xuyên sâu {crack_um:.0f} µm"),unsafe_allow_html=True)
    with s3:
        st.markdown(info_card('ĂN MÒN PHÁT TRIỂN ĐẾN NGƯỠNG',
                f"{service['propagation_years']:.1f} năm",
                f"i_corr = {i_corr:.2f} µA/cm² · giả định ăn mòn đều"),unsafe_allow_html=True)
    with s4:
        st.markdown(info_card(f'MỐC HAO HỤT {area_limit}% TIẾT DIỆN THÉP',
                f"{service['total_years']:.1f} năm" if service['total_years'] is not None else unknown_txt,
                'Không phải thời điểm phá hoại hoặc kết luận an toàn kết cấu',True),unsafe_allow_html=True)
    st.caption('Các mốc do người dùng tự chọn (độ xuyên sâu gây nứt, ngưỡng hao hụt thép). ASTM G102-23 hướng dẫn chuyển tốc độ '
               'ăn mòn điện hóa; không cung cấp thời gian hết tuổi thọ cho cấu kiện này.')
else:
    (st.error if blocked_kind == 'error' else st.info)(blocked_msg)

main_tab, service_tab, compare_tab, validate_tab, report_tab, method_tab = st.tabs([
    '📈 Diễn biến Cl⁻',
    '🏗️ Tuổi thọ sơ bộ',
    '🔎 So sánh tỷ lệ tro bay',
    '🧠 Đánh giá AI',
    '📄 Báo cáo PDF',
    '📚 Mô hình & tiêu chuẩn',
])

st.markdown('<div class="cl-foot">CloriLife AI · Concrete Durability Intelligence · '
            'Research prototype v0.4.1 · Kết quả chỉ phục vụ nghiên cứu, không thay thế thiết kế và kiểm định kết cấu</div>',
            unsafe_allow_html=True)

comparison_df = pd.DataFrame()
comparison_origin = 'Không có'
with compare_tab:
    st.markdown('<div class="cl-panel-title">So sánh 0% – 40% tro bay theo các kịch bản D28</div>',unsafe_allow_html=True)
    st.markdown('<div class="cl-tab-intro">Nhập D28 cho từng cấp phối. Giữ cố định N/CKD, lớp bảo vệ, Cs, Ccrit và m để khảo sát '
                'sự khác biệt do D28 của từng cấp phối. Không tự suy diễn quan hệ nhân quả khi thiếu dữ liệu thí nghiệm. '
                'AI chỉ dự báo RCM nên không dùng để điền D28 ở đây.</div>',unsafe_allow_html=True)
    comparison_origin = 'BẢNG D28 GIẢ ĐỊNH / DO NGƯỜI DÙNG CHỈNH SỬA'
    st.warning('Các giá trị D28 mặc định là GIẢ ĐỊNH MINH HỌA, không phải dữ liệu nghiên cứu. '
               'Hãy sửa thành kết quả thí nghiệm thực nếu muốn so sánh kỹ thuật.')
    scenarios = st.data_editor(
        DEFAULT_SCENARIOS.copy(),num_rows='dynamic',width='stretch',
        hide_index=True,key='scenario_editor',
        column_config={
            'Tro bay (%)':st.column_config.NumberColumn('Tro bay (%)',min_value=0,max_value=100,step=1,required=True),
            'D28 (x10^-12 m2/s)':st.column_config.NumberColumn('D28 (×10⁻¹² m²/s)',min_value=0.001,step=0.1,format='%.3f',required=True),
        }
    )
    comparison_df = pd.DataFrame()
    if not scenarios.empty:
        try:
            comparison_df=run_comparison(scenarios,cover,m,cs,critical,c0,horizon)
        except (ValueError, ArithmeticError) as exc:
            st.error(f'Lỗi bảng kịch bản: {exc}')
        else:
            cleft,cright=st.columns([1.15,1],gap='large')
            with cleft:
                st.markdown('**Thời gian đạt ngưỡng theo hàm lượng tro bay**')
                bars=alt.Chart(comparison_df).mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5,color='#137f90').encode(
                    x=alt.X('Tro bay (%):O',title='Tro bay (% chất kết dính)'),
                    y=alt.Y('Thời gian đạt ngưỡng (năm):Q',title='Thời gian (năm)'),
                    tooltip=['Tro bay (%):Q',alt.Tooltip('Thời gian đạt ngưỡng (năm):Q',format='.2f')]
                ).properties(height=295)
                st.altair_chart(bars,width='stretch')
            with cright:
                st.markdown('**Hệ số khuếch tán của từng cấp phối**')
                line=alt.Chart(comparison_df).mark_line(point=True,color='#0b637e',strokeWidth=3).encode(
                    x=alt.X('Tro bay (%):Q',title='Hàm lượng tro bay (%)'),
                    y=alt.Y('D28 (x10^-12 m2/s):Q',title='D28 (×10⁻¹² m²/s)'),
                    tooltip=['Tro bay (%):Q',alt.Tooltip('D28 (x10^-12 m2/s):Q',format='.3f')]
                ).properties(height=295)
                st.altair_chart(line,width='stretch')
            st.dataframe(comparison_df,hide_index=True,width='stretch',
                         column_config={'Thời gian đạt ngưỡng (năm)':st.column_config.NumberColumn(format='%.2f')})
            st.caption(f'Nguồn so sánh: {comparison_origin}. Trống nghĩa là chưa đạt ngưỡng trong {horizon} năm.')
            st.download_button('⬇ Tải bảng so sánh CSV',
                comparison_df.to_csv(index=False).encode('utf-8-sig'),
                file_name='CloriLife_v04_fly_ash_comparison.csv',mime='text/csv')

with validate_tab:
    st.markdown('<div class="cl-panel-title">AI dự báo hệ số RCM (di trú nhanh)</div>',unsafe_allow_html=True)
    st.markdown('<div class="cl-tab-intro">AI học từ dữ liệu RCM của tài liệu, KHÔNG phải D28 khuếch tán tự nhiên. '
                'Kết quả chỉ dùng để khảo sát xu hướng. Không đưa vào mô hình Fick và không tính được tuổi thọ.</div>',unsafe_allow_html=True)
    if training_bytes is None:
        st.info('Chưa có dữ liệu AI. Chọn nguồn dữ liệu ở thanh bên trái.')
    else:
        dh = hashlib.md5(training_bytes).hexdigest()[:12]
        holdout_store = st.session_state.setdefault('holdout_runs', {})
        cv_store = st.session_state.setdefault('cv_runs', {})

        st.markdown('**1. Huấn luyện từng mô hình (holdout 80/20)**')
        st.caption('Mỗi nút huấn luyện một mô hình trên 80% dữ liệu và đo trên 20% giữ lại. '
                   'Dùng để xem nhanh; KHÔNG nên chọn mô hình chỉ dựa trên bảng này.')
        btn_cols = st.columns(len(MODEL_NAMES))
        clicked = None
        for col, name in zip(btn_cols, MODEL_NAMES):
            with col:
                if st.button(f'▶ {SHORT_NAMES[name]}', key=f'run_{name}_{dh}'):
                    clicked = name
        run_all = st.button('▶ Chạy tất cả mô hình', key=f'run_all_{dh}')
        todo = list(MODEL_NAMES) if run_all else ([clicked] if clicked else [])
        for name in todo:
            with st.spinner(f'Đang huấn luyện {SHORT_NAMES[name]}...'):
                _, mt, _ = train_from_csv(training_bytes, name)
            holdout_store[(dh, name)] = mt
        rows = []
        for name in MODEL_NAMES:
            mt = holdout_store.get((dh, name))
            rows.append({
                'Mô hình': SHORT_NAMES[name],
                'Trạng thái': 'Đã huấn luyện' if mt else 'Chưa chạy',
                'R² (log₁₀)': round(mt['R2_log10'], 3) if mt else None,
                'MAE (×10⁻¹²)': round(mt['MAE_1e12'], 2) if mt else None,
                'RMSE (×10⁻¹²)': round(mt['RMSE_1e12'], 2) if mt else None,
                'Mẫu kiểm tra': mt['test_count'] if mt else None,
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch')

        st.markdown('**2. So sánh công bằng bằng CV lặp (cơ sở để chọn)**')
        st.caption('CV theo tài liệu gốc, 5 lần lặp × 5 fold trên toàn bộ dữ liệu. Mô hình được chọn theo kết quả này. '
                   'Chọn theo holdout sẽ làm kết quả báo cáo bị lạc quan.')
        if st.button('▶ Chạy CV lặp để so sánh', key=f'run_cv_{dh}'):
            with st.spinner('Đang chạy CV lặp, có thể mất vài phút (GPR chậm nhất)...'):
                cv_store[dh] = repeated_cv(pd.read_csv(BytesIO(training_bytes)))
        if dh in cv_store:
            cvdf = cv_store[dh].copy()
            cvdf['Mô hình'] = cvdf['model'].map(SHORT_NAMES)
            cvdf = cvdf.rename(columns={'R2_log10_mean': 'R² CV trung bình (log₁₀)',
                                        'R2_log10_std': 'Độ lệch chuẩn'})
            st.dataframe(cvdf[['Mô hình', 'R² CV trung bình (log₁₀)', 'Độ lệch chuẩn']],
                         hide_index=True, width='stretch')
            st.caption(f"Đứng đầu CV: {cvdf.iloc[0]['Mô hình']}. Nếu hai mô hình chênh nhau dưới 0,02 R², "
                       'coi như ngang nhau và nên chọn mô hình đơn giản hơn.')
        else:
            st.caption('Chưa chạy CV.')

        st.markdown('**3. Chọn mô hình dùng cho dự báo**')
        active = st.selectbox('Mô hình đang dùng', list(MODEL_NAMES),
                              index=list(MODEL_NAMES).index(DEFAULT_MODEL),
                              format_func=lambda k: MODEL_NAMES[k], key='active_model')
        try:
            with st.spinner('Đang huấn luyện mô hình đã chọn...'):
                model, metrics, bounds = train_from_csv(training_bytes, active)
            st.success(f"Đang dùng {MODEL_NAMES[active]}: {metrics['train_count']} mẫu huấn luyện · "
                       f"{metrics['test_count']} mẫu kiểm tra")
        except Exception as exc:
            st.error(f'Không thể huấn luyện: {exc}')

    if metrics is None:
        st.info('Chọn mô hình ở trên để xem kết quả chi tiết.')
    else:
        st.caption(f"Mô hình: {metrics['model_label']} · Nguồn: {data_label} · Đại lượng: {metrics['target']}")
        for warn_text in metrics.get('warnings',[]):
            st.warning(warn_text)
        cols=st.columns(4,gap='medium')
        for cell,(label,value,foot) in zip(cols,[
            ('MAE (×10⁻¹² m²/s)',f"{metrics['MAE_1e12']:.2f}",f"Mốc cơ sở: {metrics['baseline_MAE_1e12']:.2f}"),
            ('RMSE (×10⁻¹² m²/s)',f"{metrics['RMSE_1e12']:.2f}",'Tập kiểm tra giữ lại'),
            ('R² (log₁₀ RCM)',f"{metrics['R2_log10']:.3f}",'Không phải tỷ lệ % chính xác'),
            ('Mẫu kiểm tra',f"{metrics['test_count']} / {metrics['n']}",f"{metrics['groups_total']} tài liệu gốc"),
        ]):
            with cell: st.markdown(info_card(label,value,foot),unsafe_allow_html=True)
        st.caption(f"Phân chia: {metrics['split_strategy']}. Đã loại {metrics['filtered_rows']} dòng trùng lặp hoặc không hợp lệ.")
        left,right=st.columns([1.1,0.9],gap='large')
        with left:
            st.markdown('**RCM thực đo và AI dự báo (tập kiểm tra)**')
            observed=metrics['holdout'].rename(columns={'D_RCM_thuc_te':'Thực đo','D_RCM_AI':'AI dự báo'})
            lo=float(min(observed['Thực đo'].min(),observed['AI dự báo'].min()))
            hi=float(max(observed['Thực đo'].max(),observed['AI dự báo'].max()))
            diag=alt.Chart(pd.DataFrame({'Thực đo':[lo,hi],'AI dự báo':[lo,hi]})).mark_line(color='#dca459',strokeDash=[5,5]).encode(
                x=alt.X('Thực đo:Q',title='RCM thực đo (×10⁻¹² m²/s)'),
                y=alt.Y('AI dự báo:Q',title='RCM AI dự báo (×10⁻¹² m²/s)'))
            pts=alt.Chart(observed).mark_circle(color='#147f96',size=95,opacity=0.75).encode(
                x=alt.X('Thực đo:Q',title='RCM thực đo (×10⁻¹² m²/s)'),
                y=alt.Y('AI dự báo:Q',title='RCM AI dự báo (×10⁻¹² m²/s)'),
                tooltip=[alt.Tooltip('Thực đo:Q',format='.2f'),alt.Tooltip('AI dự báo:Q',format='.2f')])
            st.altair_chart((diag+pts).properties(height=340),width='stretch')
        with right:
            st.markdown('**Mức quan trọng đặc trưng (hoán vị)**')
            fi=metrics['importance'].copy()
            st.altair_chart(alt.Chart(fi).mark_bar(color='#237b90').encode(
                x=alt.X('importance:Q',title='Giảm R² khi xáo trộn đặc trưng (không phải quan hệ nhân quả)'),
                y=alt.Y('feature:N',sort='-x',title=None),
                tooltip=[alt.Tooltip('feature:N'),alt.Tooltip('importance:Q',format='.3f')]
            ).properties(height=340),width='stretch')
        st.download_button('⬇ Tải bảng RCM thực đo – AI dự báo (CSV)',
            metrics['holdout'].to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_v04_rcm_validation.csv',mime='text/csv')

    st.markdown('<div class="cl-section">KHẢO SÁT THEO TỶ LỆ TRO BAY (AI)</div>',unsafe_allow_html=True)
    if model is None:
        st.info('Cần huấn luyện AI ở thanh bên trái để khảo sát.')
    else:
        sweep_rows=[]
        for rate in range(0,41,5):
            vals=dict(ai_base); vals['fly_ash_fraction_of_binder']=rate/100
            if vals['fly_ash_fraction_of_binder']+vals['slag_fraction_of_binder']>1:
                continue
            value,outside=predict(model,vals,bounds)
            sweep_rows.append({'Tro bay (%)':rate,
                               'RCM AI (×10⁻¹² m²/s)':None if outside else round(value,3),
                               'Ngoài miền huấn luyện':', '.join(outside) if outside else '—'})
        sweep=pd.DataFrame(sweep_rows)
        st.dataframe(sweep,hide_index=True,width='stretch')
        plotted=sweep.dropna(subset=['RCM AI (×10⁻¹² m²/s)'])
        if plotted.empty:
            st.warning('Không có tỷ lệ tro bay nào nằm trong miền huấn luyện với các thông số hiện tại.')
        else:
            st.altair_chart(alt.Chart(plotted).mark_line(point=True,color='#0b637e',strokeWidth=3).encode(
                x=alt.X('Tro bay (%):Q',title='Tro bay (% chất kết dính)'),
                y=alt.Y('RCM AI (×10⁻¹² m²/s):Q',title='RCM AI (×10⁻¹² m²/s)'),
                tooltip=['Tro bay (%):Q',alt.Tooltip('RCM AI (×10⁻¹² m²/s):Q',format='.2f')]
            ).properties(height=295),width='stretch')
        st.caption('Giữ nguyên N/CKD, xỉ, họ và cấp xi măng, phương pháp thử theo thanh bên. Tỷ lệ ngoài miền huấn luyện bị bỏ qua, không ngoại suy. '
                   'Đây là RCM, không phải D28, nên không tính được tuổi thọ từ kết quả này.')
        st.download_button('⬇ Tải khảo sát RCM theo tro bay (CSV)',
            sweep.to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_v04_rcm_flyash_sweep.csv',mime='text/csv')

with method_tab:
    st.markdown('<div class="cl-panel-title">Cơ sở khoa học và kiểm soát chất lượng dữ liệu</div>',unsafe_allow_html=True)
    st.latex(r'D(t)=D_{28}\left(\frac{t_{28}}{t}\right)^m\quad (t>28\,\mathrm{ngày})')
    st.latex(r'C(x,t)=C_0+(C_s-C_0)\,\mathrm{erfc}\left(\frac{x}{2\sqrt{\int_0^t D(s)\,ds}}\right)')
    st.markdown('**Mô hình:** khuếch tán một chiều trong môi trường bán vô hạn, hệ số khuếch tán biến thiên theo tuổi, '
                'nồng độ clorua bề mặt không đổi. Trước 28 ngày dùng D28 cố định. Giai đoạn 2 dùng tốc độ ăn mòn đều theo i_corr giả định, với giới hạn hao hụt tiết diện do người dùng đặt.')
    st.markdown('**Mốc nứt lớp bảo vệ:** thời gian từ khởi phát đến khi độ xuyên sâu thép đạt giá trị giả định do người dùng đặt (µm), '
                'tính bằng độ xuyên sâu chia tốc độ ăn mòn. Mốc này thường đến sớm hơn nhiều so với mốc hao hụt tiết diện vài phần trăm, '
                'nhưng giá trị gây nứt phụ thuộc lớp bảo vệ, đường kính và chất lượng bê tông nên cần hiệu chuẩn.')
    st.markdown('**Bộ dữ liệu AI (RCM):** các cột `reference`, `cement`, `method`, `fly_ash_fraction_of_binder`, '
                '`slag_fraction_of_binder`, `water_binder_ratio`, `D_RCM_1e-12_m2_per_s`. Đại lượng là hệ số di trú nhanh (RCM), '
                '**không phải D28**. Không trộn trực tiếp với hệ số D28 hay hệ số theo ASTM C1556. '
                'Các mẫu cùng tài liệu gốc luôn nằm cùng một phía khi chia tập kiểm tra.')
    st.markdown('**Tiêu chuẩn tham khảo:** TCVN 12041:2017 (độ bền lâu), TCVN 9346:2012 (chống ăn mòn môi trường biển), '
                'TCVN 10302:2014 (tro bay), ASTM C1556 (khuếch tán clorua) và ASTM G102-23 (chuyển dòng ăn mòn thành tốc độ mất thép). Đây là căn cứ nghiên cứu, '
                '**không có nghĩa phần mềm đã được chứng nhận phù hợp tiêu chuẩn**.')
    st.markdown('**Độ chính xác AI:** giữ lại 20% theo tài liệu gốc; CV lặp trên toàn bộ dữ liệu dùng để chọn mô hình mặc định. '
                'MAE, RMSE có đơn vị ×10⁻¹² m²/s; R² có thể âm và không phải tỷ lệ % chính xác. '
                'Cần kiểm định trên dữ liệu độc lập trước khi dùng kết quả cho kỹ thuật.')
    st.markdown('**Giới hạn:** chưa xét liên kết clorua, hút mao dẫn, xâm nhập đối lưu, '
                'ướt–khô theo chu kỳ, ăn mòn rỗ, nứt và bong lớp bảo vệ, nhiệt–ẩm biến đổi, '
                'độ tin cậy hay khả năng chịu lực còn lại của kết cấu.')
    template = pd.DataFrame(columns=REQUIRED)
    st.download_button('⬇ Tải mẫu cấu trúc CSV trống',
                       template.to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_AI_training_template.csv',mime='text/csv')

if not ready:
    for _tab in (main_tab, service_tab, report_tab):
        with _tab:
            st.info('Cần D28 hợp lệ và C0 < Ccrit < Cs (xem phần đầu trang) để hiển thị nội dung này.')

if ready:
    with main_tab:
        st.markdown('<div class="cl-panel-title">Diễn biến nồng độ clorua tại vị trí cốt thép</div>',unsafe_allow_html=True)
        st.markdown('<div class="cl-panel-sub">Đường xanh: nồng độ Cl⁻ dự báo; đường đứt nét: Ccrit giả định</div>',unsafe_allow_html=True)
        series = pd.DataFrame({'Năm':time_axis,'Clorua tại cốt thép (%)':chloride_axis})
        main_curve = alt.Chart(series).mark_line(color='#147f96',strokeWidth=3).encode(
            x=alt.X('Năm:Q',title='Thời gian phân tích (năm)',scale=alt.Scale(domain=[0,horizon])),
            y=alt.Y('Clorua tại cốt thép (%):Q',title='Cl⁻ (% khối lượng chất kết dính)'),
            tooltip=[alt.Tooltip('Năm:Q',format='.1f'),alt.Tooltip('Clorua tại cốt thép (%):Q',format='.3f')]
        )
        threshold = alt.Chart(pd.DataFrame({'Clorua tại cốt thép (%)':[critical]})).mark_rule(color='#d49a46',strokeDash=[6,5],strokeWidth=2).encode(
            y=alt.Y('Clorua tại cốt thép (%):Q',title='Cl⁻ (% khối lượng chất kết dính)'))
        st.altair_chart((main_curve+threshold).properties(height=350,background='#ffffff').configure_axis(
            gridColor='#e9eef4',domainColor='#ccd7e0',labelColor='#597287',titleColor='#27485e'
        ).configure_view(stroke=None),width='stretch')
        if years is None:
            st.info(f'Chưa đạt Ccrit trong {horizon} năm phân tích. Không thể suy ra tuổi thọ thực > {horizon} năm.')
        else:
            st.success(f'Mô hình tính đạt Ccrit sau {years:.1f} năm với bộ giả định hiện tại.')
        st.download_button('⬇ Tải đường cong nồng độ CSV',
            data=series.to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_v04_chloride_curve.csv',mime='text/csv')

    with service_tab:
        st.markdown('<div class="cl-panel-title">Tổng hợp hai giai đoạn: khởi phát và ăn mòn phát triển</div>',unsafe_allow_html=True)
        st.markdown('<div class="cl-tab-intro">Từ thời điểm clorua đạt ngưỡng, giả định thép ăn mòn ĐỀU với mật độ dòng i_corr không đổi. '
                    'Hai mốc kết thúc do người dùng định nghĩa: độ xuyên sâu gây nứt lớp bảo vệ (µm) và tỷ lệ hao hụt diện tích '
                    'tiết diện thanh thép. Cả hai đều không phải trạng thái giới hạn chịu lực được xác minh.</div>', unsafe_allow_html=True)
        cols=st.columns(4)
        cols[0].metric('Đường kính thép',f'{bar_diameter:.1f} mm')
        cols[1].metric('Mật độ dòng ăn mòn',f'{i_corr:.2f} µA/cm²')
        cols[2].metric('Tốc độ xuyên sâu',f"{service['rate_mm_per_year']:.4f} mm/năm")
        cols[3].metric('Độ xuyên sâu tại ngưỡng hao hụt',f"{service['penetration_limit_mm']:.3f} mm")
        crack_cols=st.columns(3)
        crack_cols[0].metric('Độ xuyên sâu gây nứt (giả định)',f'{crack_um:.0f} µm')
        crack_cols[1].metric('Từ khởi phát đến nứt',f"{service['cracking_years']:.1f} năm")
        crack_cols[2].metric('Hao hụt tiết diện khi nứt',f"{service['cracking_area_loss_pct']:.2f} %")
        st.latex(r'v_{corr} \approx 0.0116 \, i_{corr} \quad [\mathrm{mm/year}]')
        st.latex(r'A_{remain}/A_0 = (1-2p/d_0)^2, \quad T_{limit}=T_{init}+p_{limit}/v_{corr}, \quad T_{crack}=T_{init}+p_{crack}/v_{corr}')
        if service['cracking_years'] < service['propagation_years']:
            st.info(f"Mốc nứt ({service['cracking_years']:.1f} năm sau khởi phát) đến sớm hơn mốc hao hụt {area_limit}% "
                    f"({service['propagation_years']:.1f} năm). Khi lớp bảo vệ đã nứt hoặc bong, điều kiện ẩm/oxy và i_corr có thể "
                    'thay đổi nên phần diễn biến phía sau mốc nứt kém tin cậy hơn.')
        if service['total_years'] is None:
            st.warning('Trong giới hạn phân tích đã chọn, Cl⁻ chưa đạt ngưỡng khởi phát. '
                       'Vì vậy KHÔNG suy diễn một thời điểm nứt hoặc hao hụt tiết diện cụ thể.')
        else:
            st.markdown(f"**Ví dụ hiện tại:** đạt ngưỡng Cl⁻ sau {years:.1f} năm; "
                        f"mốc nứt giả định sau **{service['cracking_total_years']:.1f} năm**; "
                        f"mốc hao hụt {area_limit}% tiết diện sau **{service['total_years']:.1f} năm**.")
            trajectory = remaining_bar_history(service)
            x_enc = alt.X(f'{COL_TIME}:Q',title='Thời gian (năm)')
            y_enc = alt.Y(f'{COL_REMAIN}:Q',title='Tiết diện thanh thép còn lại (%)',scale=alt.Scale(domain=[0,105]))
            chart = alt.Chart(trajectory).mark_line(color='#0c8194',strokeWidth=3).encode(
                x=x_enc, y=y_enc,
                tooltip=[alt.Tooltip(f'{COL_TIME}:Q',format='.1f'),
                         alt.Tooltip(f'{COL_REMAIN}:Q',format='.2f')]
            ).properties(height=305)
            limit_y = alt.Chart(pd.DataFrame({COL_REMAIN:[100-area_limit]})).mark_rule(
                color='#d49a46',strokeDash=[6,5],strokeWidth=2).encode(y=y_enc)
            crack_x = alt.Chart(pd.DataFrame({COL_TIME:[service['cracking_total_years']]})).mark_rule(
                color='#c2503f',strokeDash=[2,4],strokeWidth=2).encode(x=x_enc)
            st.altair_chart(chart+limit_y+crack_x,width='stretch')
            st.caption('Đường cam: ngưỡng hao hụt tiết diện đã chọn · Đường đỏ đứt: mốc nứt lớp bảo vệ (giả định).')
            st.download_button('⬇ Tải diễn biến tiết diện thép (CSV)',
                trajectory.to_csv(index=False).encode('utf-8-sig'),
                'CloriLife_v04_steel_area_curve.csv',mime='text/csv')
        st.markdown('**Độ nhạy theo giả định tốc độ ăn mòn**')
        scenarios_corr = current_density_scenarios(years,bar_diameter,area_limit,
                  currents=sorted(set([0.25,0.5,1.0,2.0,5.0,round(float(i_corr),3)])),
                  cracking_penetration_um=crack_um)
        st.dataframe(scenarios_corr,hide_index=True,width='stretch',
                     column_config={
                         COL_ICORR:st.column_config.NumberColumn(format='%.2f'),
                         COL_T_CRACK:st.column_config.NumberColumn(format='%.1f'),
                         COL_T_PROP:st.column_config.NumberColumn(format='%.1f'),
                         COL_T_LIMIT:st.column_config.NumberColumn(format='%.1f'),
                     })
        st.download_button('⬇ Tải bảng độ nhạy tốc độ ăn mòn (CSV)',
            scenarios_corr.to_csv(index=False).encode('utf-8-sig'),
            'CloriLife_v04_corrosion_sensitivity.csv',mime='text/csv')
        st.warning('GIỚI HẠN KHOA HỌC: Đây là chỉ số theo kịch bản với i_corr nhập vào, không phải AI '
                   'dự báo i_corr. i_corr được giữ như nhau cho mọi cấp phối, nên lợi ích của tro bay ở giai đoạn ăn mòn '
                   '(điện trở suất cao hơn) chưa được tính. Chưa xét ăn mòn rỗ cục bộ (mất tiết diện tại điểm rỗ có thể '
                   'lớn hơn nhiều giá trị trung bình), oxy/ẩm, nứt bê tông, bong lớp bảo vệ, độ bám dính, bố trí thép hoặc '
                   'mô hình sức chịu tải. Không dùng để quyết định thiết kế.')

    with report_tab:
        st.markdown('<div class="cl-panel-title">Xuất báo cáo nghiên cứu PDF</div>',unsafe_allow_html=True)
        st.markdown('<div class="cl-tab-intro">Báo cáo A4 ba trang gồm thông số, đường cong Cl⁻, mốc nứt và hao hụt thép, so sánh cấp phối và giới hạn khoa học.</div>',unsafe_allow_html=True)
        if comparison_df.empty:
            st.caption('Chưa có bảng so sánh hợp lệ. Mở tab “So sánh tỷ lệ tro bay” để thiết lập kịch bản trước khi xuất báo cáo.')
        if comparison_origin == 'BẢNG D28 GIẢ ĐỊNH / DO NGƯỜI DÙNG CHỈNH SỬA' and not comparison_df.empty:
            st.warning('Bảng so sánh sử dụng giá trị nhập thủ công và có thể là giả định. Cần xác nhận D28 thí nghiệm trước khi công bố.')
        params=dict(
            fly_ash=fly_ash,wb=wb,cover=cover,cs=cs,c0=c0,critical=critical,
            horizon=horizon,d28=d28,m=m,years=years,
            d_source='D28 nhập thủ công',d_source_en='Manual D28 input',
            model_source=(data_label if training_bytes is not None else 'Không dùng AI'),
            model_source_en=(data_label_en if training_bytes is not None else 'AI not used'),
            synthetic=False,comparison_origin=comparison_origin,
            bar_diameter=bar_diameter, i_corr=i_corr, area_limit=area_limit,
        )
        try:
            pdf=build_pdf_report(params,time_axis,chloride_axis,
                                 comparison=comparison_df if not comparison_df.empty else None,
                                 validation=metrics, propagation=service)
            st.download_button('⬇ TẢI BÁO CÁO CLORILIFE AI v0.4.1 (PDF)',
                               data=pdf,file_name='CloriLife_AI_v04_Research_Report.pdf',
                               mime='application/pdf',type='primary',width='stretch')
            st.success('Báo cáo PDF đã được tạo từ thông số và kết quả hiện tại.')
        except Exception as exc:
            st.error(f'Không thể tạo PDF: {exc}')
