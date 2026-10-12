"""CloriLife AI - research MVP for chloride-induced corrosion initiation.

The tool estimates threshold arrival time under a screening model, NOT
structural service life or safety compliance.
"""

from html import escape
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st

from physics import concentration, initiation_years
from ml import FEATURES, fit_model, predict


st.set_page_config(
    page_title="CloriLife AI | Concrete Durability Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom presentation only; the physics and ML engines are unchanged.
st.markdown(
    """
<style>
    /* Layout and typography */
    [data-testid="stAppViewContainer"] {background: #f5f8fc;}
    .main .block-container {max-width: 1420px; padding-top: 1.25rem; padding-bottom: 2.8rem;}
    [data-testid="stSidebar"] {background: #edf3f8; border-right: 1px solid #dbe4ed;}
    [data-testid="stSidebar"] .block-container {padding-top: 1.3rem;}
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {color: #183b55;}
    [data-testid="stMetricValue"] {font-weight: 750; color: #173a54;}
    div[data-testid="stAlert"] {border-radius: 12px;}
    [data-testid="stForm"] {border-radius: 14px;}
    .stTabs [data-baseweb="tab-list"] {gap: 18px;}
    .stTabs [data-baseweb="tab"] {font-weight: 700;}
    button[kind="primary"] {background-color: #137e91; border-color: #137e91;}

    /* Brand hero */
    .cl-hero {
        position: relative; overflow: hidden; border-radius: 19px;
        padding: 30px 36px 32px 36px;
        background: radial-gradient(circle at 89% 18%, rgba(79,213,206,.21), transparent 28%),
                    linear-gradient(108deg, #102741 0%, #173e5d 63%, #126c79 100%);
        box-shadow: 0 12px 28px rgba(25,55,81,.12);
        color: white; margin-bottom: 16px;
    }
    .cl-hero:after {
        content: ""; position: absolute; height: 230px; width: 230px;
        border-radius: 50%; border: 1px solid rgba(176,245,235,.20);
        right: -56px; top: -100px; box-shadow: 0 0 0 40px rgba(176,245,235,.04), 0 0 0 84px rgba(176,245,235,.03);
    }
    .cl-eyebrow {letter-spacing: 2.2px; color: #9ee1da; font-size: 11px; font-weight: 800;}
    .cl-title {font-size: clamp(34px, 3.5vw, 57px); line-height: 1.05; font-weight: 850;
        letter-spacing: -1.5px; color: #fff; margin: 13px 0 12px 0;}
    .cl-title span {color: #70e3d2;}
    .cl-description {max-width: 900px; font-size: 15px; line-height: 1.65;
        color: #e4edf4; margin-bottom: 19px;}
    .cl-pill {border: 1px solid rgba(210,244,239,.35); background: rgba(240,252,250,.09);
        border-radius: 30px; padding: 6px 12px; font-size: 11px; color: #def4f2;
        display: inline-block; margin-right: 7px; margin-bottom: 3px; font-weight: 700;}
    .cl-sidebar-brand {padding: 12px 13px 13px 13px; background: #163b59; color: #fff;
        border-radius: 13px; margin-bottom: 19px;}
    .cl-sidebar-brand strong {font-size: 19px; letter-spacing: -.3px;}
    .cl-sidebar-brand strong span {color: #7fe3d6;}
    .cl-sidebar-brand small {display: block; color: #b9d2df; margin-top: 4px; font-size: 11px;}
    .cl-section {font-size: 12px; letter-spacing: 1.3px; text-transform: uppercase;
        color: #337187; font-weight: 850; margin: 12px 0 8px 0;}
    .cl-kpi {background: #fff; border: 1px solid #e2eaf0; border-radius: 15px;
        padding: 18px 19px; min-height: 119px; box-shadow: 0 3px 12px rgba(17,52,75,.035);}
    .cl-kpi-label {font-size: 12px; color: #617a8a; font-weight: 700; line-height: 1.4;}
    .cl-kpi-value {font-size: clamp(24px, 2.3vw, 37px); color: #133b58;
        font-weight: 850; line-height: 1.3; letter-spacing: -0.8px; margin: 4px 0;}
    .cl-kpi-value.teal {color: #047b80;}
    .cl-kpi-sub {font-size: 11px; color: #728d9a;}
    .cl-note {border-left: 4px solid #d59b45; padding: 13px 17px; background: #fff8e9;
        border-radius: 0 11px 11px 0; font-size: 13px; line-height: 1.65;
        color: #65502c; margin: 12px 0 20px 0;}
    .cl-panel-title {font-size: 20px; color: #173a54; font-weight: 850;
        margin: 12px 0 3px 0;}
    .cl-panel-sub {color: #5e788a; font-size: 13px; margin-bottom: 16px;}
    .cl-info-panel {background: white; border: 1px solid #e3edf3;
        border-radius: 14px; padding: 18px; font-size: 13px; line-height: 1.72; color: #3b5667;}
    .cl-info-panel strong {color: #173b57;}
    .cl-foot {color: #8293a0; font-size: 11px; text-align: center;
        padding: 21px 12px 2px;}
</style>
""",
    unsafe_allow_html=True,
)


def info_card(label, value, note, emphasized=False):
    """Display human-readable numeric inputs only in a visual KPI card."""
    color_class = "teal" if emphasized else ""
    return (
        '<div class="cl-kpi">'
        f'<div class="cl-kpi-label">{escape(str(label))}</div>'
        f'<div class="cl-kpi-value {color_class}">{escape(str(value))}</div>'
        f'<div class="cl-kpi-sub">{escape(str(note))}</div>'
        '</div>'
    )


with st.sidebar:
    st.markdown(
        '<div class="cl-sidebar-brand"><strong>CloriLife <span>AI</span></strong>'
        '<small>CONCRETE DURABILITY LAB · v0.2</small></div>',
        unsafe_allow_html=True,
    )
    st.markdown('**01 — Nguồn dữ liệu khuếch tán**')
    mode = st.radio(
        'Phương pháp xác định D28',
        ['Nhập D28 đo được / tham khảo', 'Dự báo D28 bằng AI từ CSV'],
        index=0,
        help='Chỉ sử dụng AI khi có tập CSV phù hợp và được kiểm chứng.',
    )
    model = None
    bounds = None
    metrics = None
    if mode == 'Dự báo D28 bằng AI từ CSV':
        uploaded = st.file_uploader('Tải bộ dữ liệu huấn luyện (.csv)', type=['csv'])
        if uploaded is not None:
            try:
                source = pd.read_csv(uploaded)
                model, metrics, bounds = fit_model(source)
                st.success(f"Huấn luyện {metrics['n']} mẫu; MAE: {metrics['MAE_m2_s']:.2e} m²/s")
                st.caption(f"R² trên tập kiểm tra nội bộ (log10): {metrics['R2_log10']:.3f}")
            except Exception as error:
                st.error(f'Không thể huấn luyện mô hình: {error}')
        else:
            st.info('Hãy tải CSV thực nghiệm để kích hoạt dự báo AI. Không tự giả lập kết quả AI.')

    st.divider()
    st.markdown('**02 — Vật liệu và cấp phối**')
    fly_ash = st.number_input('Hàm lượng tro bay (% chất kết dính)', 0.0, 80.0, 25.0, 1.0)
    wb = st.number_input('Tỷ lệ nước / chất kết dính', 0.15, 0.90, 0.40, 0.01)
    curing = st.number_input('Tuổi mẫu khi thử (ngày)', 1, 3650, 28)
    strength = st.number_input('Cường độ chịu nén (MPa)', 5.0, 150.0, 40.0, 1.0)
    temp = st.number_input('Nhiệt độ tham chiếu (°C)', 0.0, 60.0, 25.0, 1.0)

    if mode == 'Dự báo D28 bằng AI từ CSV':
        if model is not None:
            values = dict(zip(FEATURES, [fly_ash, wb, curing, strength, temp]))
            d28, outside = predict(model, values, bounds)
            if outside:
                st.error('Ngoài miền dữ liệu huấn luyện: ' + ', '.join(outside))
            else:
                st.success(f'D28 do AI dự báo: {d28:.3e} m²/s')
        else:
            d28 = None
    else:
        d28 = st.number_input(
            'Hệ số D28 (×10⁻¹² m²/s)',
            min_value=0.01, max_value=100.0, value=8.0, step=0.5,
            help='Phải là hệ số phù hợp với mô hình khuếch tán, cùng cơ sở đơn vị và tuổi tham chiếu.',
        ) * 1e-12

    m = st.number_input('Số mũ già hóa m (hiệu chuẩn / giả định)', 0.0, 0.95, 0.35, 0.05)
    st.caption('Đầu vào mặc định chỉ dùng để minh họa, chưa được hiệu chuẩn thực tế.')


st.markdown(
    '''<div class="cl-hero">
        <div class="cl-eyebrow">APPLIED MACHINE LEARNING  /  DURABILITY ENGINEERING</div>
        <div class="cl-title">CloriLife <span>AI</span></div>
        <div class="cl-description">Nền tảng phân tích xâm nhập ion clorua và dự báo thời gian
        khởi phát ăn mòn cốt thép trong bê tông sử dụng tro bay.
        Kết hợp mô hình học máy với lý thuyết khuếch tán để hỗ trợ nghiên cứu độ bền lâu.</div>
        <span class="cl-pill">BÊ TÔNG TRO BAY</span>
        <span class="cl-pill">MÔ HÌNH KHUẾCH TÁN FICK</span>
        <span class="cl-pill">MACHINE LEARNING</span>
        <span class="cl-pill">RESEARCH MVP · v0.2</span>
    </div>''',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="cl-note"><b>Phạm vi khoa học:</b> Công cụ dự báo thời điểm nồng độ Cl⁻ '
    'tại cốt thép đạt ngưỡng giả định, <b>không phải tuổi thọ chịu lực của công trình</b>. '
    'Chưa xét nứt, phát triển ăn mòn, suy giảm tiết diện thép và độ tin cậy kết cấu.</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="cl-section">03 / KẾT CẤU VÀ MÔI TRƯỜNG</div>', unsafe_allow_html=True)
a, b, c, d = st.columns(4, gap='medium')
with a:
    cover = st.number_input('Lớp bê tông bảo vệ (mm)', 5.0, 200.0, 50.0, 1.0)
with b:
    cs = st.number_input('Clorua bề mặt Cs (%)', 0.01, 10.0, 1.20, 0.10)
with c:
    c0 = st.number_input('Clorua ban đầu C0 (%)', 0.00, 10.0, 0.00, 0.05)
with d:
    critical = st.number_input('Ngưỡng tới hạn Ccrit (%)', 0.01, 10.0, 0.40, 0.05)

slider_col, note_col = st.columns([1, 1], gap='large')
with slider_col:
    horizon = st.slider('Giới hạn thời gian phân tích (năm)', 10, 200, 100, 5)
with note_col:
    st.caption('Cs, C0, Ccrit phải cùng cơ sở khối lượng chất kết dính. Giá trị ngưỡng phải được chọn theo cơ sở nghiên cứu phù hợp.')

st.markdown('<div class="cl-section">04 / KẾT QUẢ PHÂN TÍCH</div>', unsafe_allow_html=True)

if not (c0 < critical < cs):
    st.error('Dữ liệu không hợp lệ: cần C0 < Ccrit < Cs. Vui lòng chỉnh lại trước khi tính.')
elif d28 is None:
    st.info('Bạn đang chọn chế độ AI. Hãy tải tệp dữ liệu CSV ở thanh bên trái để huấn luyện và bắt đầu phân tích.')
else:
    try:
        years = initiation_years(cover, d28, m, cs, critical, c0, horizon)
        time_axis = np.linspace(0.001, horizon, 350)
        chloride_axis = concentration(time_axis, cover, d28, m, cs, c0)
        ten_years = min(10, horizon)
        c_10 = float(concentration(ten_years, cover, d28, m, cs, c0))

        left, center, right = st.columns([1.28, 1, 1], gap='medium')
        with left:
            if years is None:
                val, sub = 'Chưa đạt ngưỡng', f'Trong phạm vi {horizon} năm; không suy ra tuổi thọ > {horizon} năm'
            else:
                val, sub = f'{years:.1f} năm', 'Dự báo thời điểm khởi phát ăn mòn do clorua'
            st.markdown(info_card('THỜI GIAN ĐẠT NGƯỠNG CL⁻', val, sub, True), unsafe_allow_html=True)
        with center:
            st.markdown(info_card('HỆ SỐ KHUẾCH TÁN D28', f'{d28 / 1e-12:.2f} ×10⁻¹²', 'Đơn vị m²/s · tuổi tham chiếu 28 ngày'), unsafe_allow_html=True)
        with right:
            st.markdown(info_card('CL⁻ TẠI VỊ TRÍ THÉP, NĂM 10', f'{c_10:.3f}%', 'Theo cơ sở khối lượng chất kết dính'), unsafe_allow_html=True)

        tab_chart, tab_data, tab_method = st.tabs([
            '📈  Diễn biến Cl⁻', '📋  Bảng số liệu & xuất kết quả', '📖  Phương pháp & tiêu chuẩn'
        ])
        with tab_chart:
            st.markdown('<div class="cl-panel-title">Diễn biến nồng độ clorua theo thời gian</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="cl-panel-sub">Đường màu xanh: Cl⁻ tại lớp cốt thép · '
                'Đường nét đứt: ngưỡng tới hạn giả định</div>', unsafe_allow_html=True
            )
            plot_df = pd.DataFrame({'Năm': time_axis, 'Clorua tại thép': chloride_axis})
            curve = alt.Chart(plot_df).mark_line(color='#147f96', strokeWidth=3).encode(
                x=alt.X('Năm:Q', title='Thời gian phân tích (năm)', scale=alt.Scale(domain=[0, horizon])),
                y=alt.Y('Clorua tại thép:Q', title='Cl⁻ (% khối lượng chất kết dính)'),
                tooltip=[alt.Tooltip('Năm:Q', format='.1f'), alt.Tooltip('Clorua tại thép:Q', format='.3f')],
            )
            threshold = alt.Chart(pd.DataFrame({'Ccrit': [critical]})).mark_rule(
                color='#e29b50', strokeDash=[6, 5], strokeWidth=2
            ).encode(y='Ccrit:Q')
            chart = (curve + threshold).properties(height=365, background='#ffffff').configure_axis(
                gridColor='#e9eef4', domainColor='#ccd7e0', labelColor='#597287', titleColor='#27485e'
            ).configure_view(stroke=None)
            st.altair_chart(chart, use_container_width=True)
            if years is None:
                st.info('Theo thông số hiện tại, nồng độ dự báo chưa vượt Ccrit trong thời gian phân tích.')
            else:
                st.success(f'Điểm cắt ngưỡng: xấp xỉ {years:.1f} năm. Kết quả chỉ có giá trị trong giả định mô hình.')
        with tab_data:
            st.markdown('#### Số liệu phục vụ báo cáo')
            results = pd.DataFrame({
                'time_years': time_axis,
                'chloride_pct_binder': chloride_axis,
                'threshold_pct_binder': critical,
            })
            st.download_button(
                '⬇ Tải bảng kết quả CSV',
                data=results.to_csv(index=False).encode('utf-8-sig'),
                file_name='clorilife_ai_chloride_prediction.csv',
                mime='text/csv',
                use_container_width=True,
            )
            parameters = pd.DataFrame([{
                'Phương pháp D28': mode,
                'Tro bay (%)': fly_ash,
                'N/CKD': wb,
                'Tuổi mẫu (ngày)': curing,
                'D28 (m²/s)': d28,
                'Số mũ m': m,
                'Lớp bảo vệ (mm)': cover,
                'Cs (%)': cs,
                'C0 (%)': c0,
                'Ccrit (%)': critical,
                'Thời gian đạt ngưỡng (năm)': years,
            }])
            st.dataframe(parameters, hide_index=True, use_container_width=True)
            st.caption('Dữ liệu CSV xuất từ mô hình lý tưởng, không phải số liệu quan trắc thực tế.')
        with tab_method:
            st.markdown('#### Cơ sở mô hình hóa')
            st.latex(r'D(t)=D_{28}\left(\frac{t_{28}}{t}\right)^m\quad (t>28\ \mathrm{ngay})')
            st.latex(r'C(x,t)=C_0+(C_s-C_0)\,\mathrm{erfc}\left(\frac{x}{2\sqrt{\int_0^t D(s)\,ds}}\right)')
            st.markdown('''
**Mô hình vật lý:** Khuếch tán một chiều trong môi trường bán vô hạn, nồng độ bề mặt không đổi; D28 không đổi trước 28 ngày và biến thiên theo lũy thừa sau 28 ngày. Bài toán sử dụng tích phân hệ số khuếch tán theo thời gian.

**AI:** Random Forest hồi quy hệ số D28 từ bộ CSV gồm các cột `fly_ash_pct`, `water_binder_ratio`, `curing_days`, `compressive_strength_mpa`, `temperature_c`, `d28_m2_s`. Phải dùng dữ liệu có phương pháp đo và đơn vị đồng nhất. Đánh giá train/test ngẫu nhiên chỉ để thử nghiệm, không chứng minh năng lực ngoại suy.

**Các tiêu chuẩn tham khảo:** TCVN 12041:2017 (thiết kế độ bền lâu), TCVN 9346:2012 (bảo vệ chống ăn mòn trong môi trường biển), TCVN 10302:2014 (tro bay cho bê tông), ASTM C1556 (thí nghiệm khuếch tán clorua). Việc viện dẫn không đồng nghĩa phần mềm đã được chứng nhận theo tiêu chuẩn.

**Giới hạn:** Chưa xét liên kết ion clorua, hút mao dẫn, môi trường ướt/khô tuần hoàn, nứt bê tông, biến động nhiệt độ-độ ẩm, giai đoạn phát triển ăn mòn, độ tin cậy hoặc suy giảm chịu lực. Kết quả chỉ phục vụ nghiên cứu và sàng lọc sơ bộ, không dùng thay tính toán thiết kế.
''')
    except ValueError as error:
        st.error(f'Lỗi tham số vật lý: {error}')

st.markdown(
    '<div class="cl-foot">CloriLife AI · Concrete Durability Intelligence · '
    'Research prototype v0.2 · Dành cho nghiên cứu, không dùng ra quyết định thiết kế kết cấu</div>',
    unsafe_allow_html=True,
)
