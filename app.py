"""CloriLife AI v0.4.1: chloride ingress + conditional uniform-corrosion scenarios.

Extends the v0.3 prototype with a two-phase steel area-loss threshold
scenario. This is not a structural capacity or code compliance analysis.
"""
from __future__ import annotations

from html import escape
import base64
from io import BytesIO
from pathlib import Path
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from comparison import DEFAULT_SCENARIOS, ai_scenario_inputs, run_comparison
from corrosion import (serviceability_indicator, remaining_bar_history,
                       current_density_scenarios, area_loss_percent,
                       COL_TIME, COL_REMAIN, COL_ICORR, COL_T_CRACK, COL_T_PROP, COL_T_LIMIT)
from ml import FEATURES, fit_model, predict
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
def train_from_csv(raw_bytes: bytes):
    """Cache fitting; never train on the visible holdout observations."""
    df = pd.read_csv(BytesIO(raw_bytes))
    return fit_model(df)


# SIDEBAR - shared input state for all six tabs.
with st.sidebar:
    st.markdown(
        '<div class="cl-sidebar-brand"><strong>CloriLife <span>AI</span></strong>'
        '<small>CONCRETE DURABILITY LAB · v0.4.1</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown('**01 — Dữ liệu và chế độ dự báo**')
    source_choice = st.selectbox('Nguồn dữ liệu AI', [
        'Chưa sử dụng bộ dữ liệu',
        'Bộ dữ liệu DEMO mô phỏng',
        'CSV thí nghiệm của tôi',
    ], help='Bộ DEMO là dữ liệu tổng hợp, KHÔNG đại diện kết quả thí nghiệm.')
    training_bytes = None
    synthetic = False
    if source_choice == 'Bộ dữ liệu DEMO mô phỏng':
        try:
            training_bytes = Path(__file__).with_name('data_demo_SYNTHETIC.csv').read_bytes()
        except OSError:
            st.error('Không tìm thấy data_demo_SYNTHETIC.csv cùng thư mục với app.py. '
                     'Hãy tải file này lên GitHub hoặc chọn nguồn dữ liệu khác.')
        else:
            synthetic = True
            st.warning('Đang dùng dữ liệu MÔ PHỎNG. Chỉ thử nghiệm chức năng AI, không đánh giá độ chính xác khoa học.')
    elif source_choice == 'CSV thí nghiệm của tôi':
        upload = st.file_uploader('Tải CSV thí nghiệm', type=['csv'], key='training_csv')
        if upload is not None:
            training_bytes = upload.getvalue()
            st.caption('Kiểm tra đơn vị, phương pháp đo và tính độc lập của các cấp phối trước khi sử dụng.')

    model, metrics, bounds = None, None, None
    if training_bytes is not None:
        try:
            model, metrics, bounds = train_from_csv(training_bytes)
            st.success(f"Đã huấn luyện: {metrics['train_count']} mẫu · kiểm tra: {metrics['test_count']} mẫu")
        except Exception as exc:
            st.error(f'Không thể huấn luyện: {exc}')

    mode = st.radio('Hệ số khuếch tán cho bài toán chính', [
        'Nhập D28 đo được / tham khảo', 'AI dự báo D28 từ CSV',
    ], help='Chế độ nhập thủ công không dự báo D28 bằng AI.', key='main_mode')
    st.divider()
    st.markdown('**02 — Bê tông tro bay và cấp phối**')
    fly_ash = st.number_input('Tro bay / chất kết dính (%)',0.0,80.0,25.0,1.0)
    wb = st.number_input('Nước / chất kết dính',0.15,0.90,0.40,0.01)
    curing = st.number_input('Tuổi mẫu khi thí nghiệm (ngày)',1,3650,28)
    strength = st.number_input('Cường độ chịu nén (MPa)',5.0,150.0,40.0,1.0)
    temp = st.number_input('Nhiệt độ tham chiếu (°C)',0.0,60.0,25.0,1.0)
    values = dict(zip(FEATURES,[fly_ash,wb,curing,strength,temp]))

    outside = []
    if mode == 'AI dự báo D28 từ CSV':
        d28 = None
        if model is not None:
            d28_pred, outside = predict(model, values, bounds)
            if outside:
                st.error('Thông số vượt miền huấn luyện: '+', '.join(outside))
            else:
                d28 = d28_pred
                st.success(f'D28 từ AI: {d28/1e-12:.3f} ×10⁻¹² m²/s')
        else:
            st.info('Hãy chọn bộ CSV phù hợp để kích hoạt dự báo AI.')
    else:
        d28 = st.number_input(
            'D28 đo được / tham khảo (×10⁻¹² m²/s)',
            min_value=0.01,max_value=100.0,value=8.0,step=0.5
        ) * 1e-12
    m = st.number_input('Số mũ già hóa m (giả định / hiệu chuẩn)',0.0,0.95,0.35,0.05)
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
elif d28 is None:
    ready, blocked_kind = False, 'info'
    blocked_msg = ('Chưa có D28 hợp lệ cho bài toán chính. Chọn nhập D28 hoặc cung cấp CSV AI '
                   'và thông số nằm trong miền dữ liệu. Các tab So sánh, Đánh giá AI vẫn dùng được.')
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
    if synthetic and mode == 'AI dự báo D28 từ CSV':
        st.warning('KẾT QUẢ AI DỰA TRÊN DỮ LIỆU MÔ PHỎNG: không sử dụng để kết luận độ bền lâu của công trình.')
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
    st.markdown('<div class="cl-tab-intro">Giữ cố định N/CKD, lớp bảo vệ, Cs, Ccrit và m để khảo sát sự khác biệt do D28 của từng cấp phối. '
                'Không tự suy diễn quan hệ nhân quả khi thiếu dữ liệu thí nghiệm.</div>',unsafe_allow_html=True)
    cmp_mode = st.radio('Nguồn D28 cho từng hàm lượng tro bay',[
        'Nhập D28 cho từng cấp phối',
        'AI dự báo từ bộ dữ liệu CSV',
    ],horizontal=True)
    scenarios = pd.DataFrame(columns=['Tro bay (%)','D28 (x10^-12 m2/s)'])
    if cmp_mode == 'Nhập D28 cho từng cấp phối':
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
    elif model is None:
        st.info('Chọn nguồn dữ liệu CSV ở thanh bên trái để sử dụng AI trong chức năng so sánh.')
    else:
        comparison_origin = 'AI TỪ DỮ LIỆU MÔ PHỎNG' if synthetic else 'AI TỪ CSV DO NGƯỜI DÙNG CUNG CẤP'
        st.warning('AI chỉ dự báo trong miền giá trị đã xuất hiện khi huấn luyện. '
                   'Các tỷ lệ 0% hoặc 40% nằm ngoài miền này sẽ bị bỏ qua, không ngoại suy.')
        st.info('Khi đổi hàm lượng tro bay, các thông số còn lại (N/CKD, cường độ nén, tuổi, nhiệt độ) được giữ nguyên '
                'theo thanh bên trái. Thực tế cường độ thay đổi theo tro bay, nên hãy xem đây là khảo sát độ nhạy, không phải dự báo cấp phối thật.')
        scenarios, skipped = ai_scenario_inputs(model,bounds,values)
        for rate, feats in skipped:
            st.info(f'Bỏ qua tỷ lệ tro bay {rate}%: ngoài miền huấn luyện ở {", ".join(feats)}.')
        if synthetic:
            st.error('Nguồn AI hiện tại là bộ dữ liệu MÔ PHỎNG, kết quả chỉ để thử thao tác phần mềm.')
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
    st.markdown('<div class="cl-panel-title">Đánh giá mô hình AI bằng tập dữ liệu độc lập khi huấn luyện</div>',unsafe_allow_html=True)
    st.markdown('<div class="cl-tab-intro">Chỉ hiển thị khi người dùng nạp CSV; các chỉ số dựa trên tập kiểm tra giữ lại 20%, '
                'không phải độ chính xác đã được kiểm chứng liên phòng thí nghiệm.</div>',unsafe_allow_html=True)
    if metrics is None:
        st.info('Chưa có dữ liệu AI. Chọn “CSV thí nghiệm của tôi” ở thanh bên trái, hoặc DEMO để thử giao diện.')
    else:
        if synthetic:
            st.error('THỬ NGHIỆM BẰNG DỮ LIỆU TỔNG HỢP: MAE, RMSE, R² KHÔNG CHỨNG MINH ĐỘ CHÍNH XÁC TRÊN BÊ TÔNG THẬT.')
        if not metrics['groups_used']:
            st.warning('Đang chia dữ liệu ngẫu nhiên theo dòng. Nếu CSV có nhiều mẫu lặp cùng cấp phối, bổ sung cột mix_id để kiểm tra theo nhóm, tránh rò rỉ thông tin.')
        for warn_text in metrics.get('warnings',[]):
            st.warning(warn_text)
        cols=st.columns(4,gap='medium')
        for cell,(label,value,foot) in zip(cols,[
            ('MAE D28',f"{metrics['MAE_m2_s']/1e-12:.3f} ×10⁻¹²",'m²/s · tập kiểm tra'),
            ('RMSE D28',f"{metrics['RMSE_m2_s']/1e-12:.3f} ×10⁻¹²",'m²/s · tập kiểm tra'),
            ('R² (log₁₀ D28)',f"{metrics['R2_log10']:.3f}",'Không phải xác suất'),
            ('Mẫu kiểm tra',f"{metrics['test_count']} / {metrics['n']}",'Giữ lại để đánh giá'),
        ]):
            with cell: st.markdown(info_card(label,value,foot),unsafe_allow_html=True)
        st.caption(f"Phân chia: {metrics['split_strategy']}. Đã loại {metrics['filtered_rows']} dòng không hợp lệ.")
        st.caption(f"MAE mốc tham chiếu (median tập train): {metrics['baseline_MAE_m2_s']/1e-12:.3f} ×10⁻¹² m²/s. Nếu MAE của AI lớn hơn, mô hình chưa vượt mốc cơ sở.")
        cv=metrics.get('cv_r2_log10')
        if cv:
            st.caption(f"Kiểm tra chéo {cv['k']}-fold: R² (log₁₀ D28) = {cv['mean']:.3f} ± {cv['std']:.3f}. "
                       'Ổn định hơn một lần chia 80/20 khi dữ liệu ít; độ lệch chuẩn lớn nghĩa là mô hình chưa ổn định.')
        left,right=st.columns([1.1,0.9],gap='large')
        with left:
            st.markdown('**Thực đo / giá trị dữ liệu vs AI dự báo (tập kiểm tra)**')
            observed=metrics['holdout'].copy()
            observed['Thực tế (×10⁻¹²)']=observed['D28_thuc_te_m2_s']/1e-12
            observed['AI dự báo (×10⁻¹²)']=observed['D28_AI_m2_s']/1e-12
            lo=float(min(observed['Thực tế (×10⁻¹²)'].min(),observed['AI dự báo (×10⁻¹²)'].min()))
            hi=float(max(observed['Thực tế (×10⁻¹²)'].max(),observed['AI dự báo (×10⁻¹²)'].max()))
            diag=alt.Chart(pd.DataFrame({'Thực tế (×10⁻¹²)':[lo,hi],'AI dự báo (×10⁻¹²)':[lo,hi]})).mark_line(color='#dca459',strokeDash=[5,5]).encode(
                x=alt.X('Thực tế (×10⁻¹²):Q',title='D28 thực tế trong CSV (×10⁻¹² m²/s)'),
                y=alt.Y('AI dự báo (×10⁻¹²):Q',title='D28 AI dự báo (×10⁻¹² m²/s)'))
            pts=alt.Chart(observed).mark_circle(color='#147f96',size=95,opacity=0.75).encode(
                x=alt.X('Thực tế (×10⁻¹²):Q',title='D28 thực tế trong CSV (×10⁻¹² m²/s)'),
                y=alt.Y('AI dự báo (×10⁻¹²):Q',title='D28 AI dự báo (×10⁻¹² m²/s)'),
                tooltip=[alt.Tooltip('Thực tế (×10⁻¹²):Q',format='.3f'),alt.Tooltip('AI dự báo (×10⁻¹²):Q',format='.3f')]
            )
            st.altair_chart((diag+pts).properties(height=340),width='stretch')
        with right:
            st.markdown('**Mức đóng góp đặc trưng của Random Forest**')
            fi=metrics['importance'].copy()
            st.altair_chart(alt.Chart(fi).mark_bar(color='#237b90').encode(
                x=alt.X('importance:Q',title='Tầm quan trọng tương đối (không phải quan hệ nhân quả)'),
                y=alt.Y('feature:N',sort='-x',title=None),
                tooltip=[alt.Tooltip('feature:N'),alt.Tooltip('importance:Q',format='.3f')]
            ).properties(height=340),width='stretch')
        st.download_button('⬇ Tải bảng quan trắc–dự báo tập kiểm tra (CSV)',
            metrics['holdout'].to_csv(index=False).encode('utf-8-sig'),
            file_name='CloriLife_v04_ai_validation.csv',mime='text/csv')
        st.caption('Điểm phân chia có cùng cấp phối cần được nhóm lại bằng mix_id. Ngoại suy ngoài khoảng huấn luyện bị từ chối trong dự báo từng kịch bản.')

with method_tab:
    st.markdown('<div class="cl-panel-title">Cơ sở khoa học và kiểm soát chất lượng dữ liệu</div>',unsafe_allow_html=True)
    st.latex(r'D(t)=D_{28}\left(\frac{t_{28}}{t}\right)^m\quad (t>28\,\mathrm{ngày})')
    st.latex(r'C(x,t)=C_0+(C_s-C_0)\,\mathrm{erfc}\left(\frac{x}{2\sqrt{\int_0^t D(s)\,ds}}\right)')
    st.markdown('**Mô hình:** khuếch tán một chiều trong môi trường bán vô hạn, hệ số khuếch tán biến thiên theo tuổi, '
                'nồng độ clorua bề mặt không đổi. Trước 28 ngày dùng D28 cố định. Giai đoạn 2 dùng tốc độ ăn mòn đều theo i_corr giả định, với giới hạn hao hụt tiết diện do người dùng đặt.')
    st.markdown('**Mốc nứt lớp bảo vệ:** thời gian từ khởi phát đến khi độ xuyên sâu thép đạt giá trị giả định do người dùng đặt (µm), '
                'tính bằng độ xuyên sâu chia tốc độ ăn mòn. Mốc này thường đến sớm hơn nhiều so với mốc hao hụt tiết diện vài phần trăm, '
                'nhưng giá trị gây nứt phụ thuộc lớp bảo vệ, đường kính và chất lượng bê tông nên cần hiệu chuẩn.')
    st.markdown('**Bộ dữ liệu AI:** các cột `fly_ash_pct`, `water_binder_ratio`, `curing_days`, '
                '`compressive_strength_mpa`, `temperature_c`, `d28_m2_s` và tùy chọn `mix_id`. '
                'D28 phải cùng định nghĩa, phương pháp và tuổi tham chiếu trên toàn bộ dữ liệu. '
                'Không được trộn trực tiếp hệ số theo ASTM C1556 với NT BUILD 492 hay điện lượng ASTM C1202.')
    st.markdown('**Tiêu chuẩn tham khảo:** TCVN 12041:2017 (độ bền lâu), TCVN 9346:2012 (chống ăn mòn môi trường biển), '
                'TCVN 10302:2014 (tro bay), ASTM C1556 (khuếch tán clorua) và ASTM G102-23 (chuyển dòng ăn mòn thành tốc độ mất thép). Đây là căn cứ nghiên cứu, '
                '**không có nghĩa phần mềm đã được chứng nhận phù hợp tiêu chuẩn**.')
    st.markdown('**Độ chính xác AI:** giữ lại 20% quan sát; nếu có `mix_id`, chia theo nhóm cấp phối. '
                'MAE/RMSE có đơn vị m²/s; R² có thể âm và không phải tỷ lệ % chính xác. '
                'Cần kiểm định trên tập dữ liệu thực độc lập trước khi dùng kết quả phục vụ kỹ thuật.')
    st.markdown('**Giới hạn:** chưa xét liên kết clorua, hút mao dẫn, xâm nhập đối lưu, '
                'ướt–khô theo chu kỳ, ăn mòn rỗ, nứt và bong lớp bảo vệ, nhiệt–ẩm biến đổi, '
                'độ tin cậy hay khả năng chịu lực còn lại của kết cấu.')
    template = pd.DataFrame(columns=FEATURES+['d28_m2_s','mix_id'])
    st.download_button('⬇ Tải mẫu cấu trúc CSV trống',
                       template.to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_AI_training_template.csv',mime='text/csv')

if not ready:
    for _tab in (main_tab, service_tab, report_tab):
        with _tab:
            st.info('Cần D28 hợp lệ và C0 < Ccrit < Cs (xem phần đầu trang) để hiển thị nội dung này.')
    st.stop()

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
    if synthetic:
        st.warning('Báo cáo sẽ ghi rõ nguồn dữ liệu AI là MÔ PHỎNG. Không trình bày chỉ số này như kết quả thí nghiệm thật.')
    if comparison_origin == 'BẢNG D28 GIẢ ĐỊNH / DO NGƯỜI DÙNG CHỈNH SỬA' and not comparison_df.empty:
        st.warning('Bảng so sánh sử dụng giá trị nhập thủ công và có thể là giả định. Cần xác nhận D28 thí nghiệm trước khi công bố.')
    params=dict(
        fly_ash=fly_ash,wb=wb,cover=cover,cs=cs,c0=c0,critical=critical,
        horizon=horizon,d28=d28,m=m,years=years,
        d_source=('AI từ CSV' if mode=='AI dự báo D28 từ CSV' else 'D28 nhập thủ công'),
        d_source_en=('AI prediction from CSV' if mode=='AI dự báo D28 từ CSV' else 'Manual D28 input'),
        model_source=('DEMO MÔ PHỎNG' if synthetic else 'CSV do người dùng tải lên'),
        model_source_en=('SYNTHETIC DEMO' if synthetic else 'user-uploaded CSV'),
        synthetic=synthetic,comparison_origin=comparison_origin,
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

