"""Optional Streamlit interface for labeled, observed time-to-limit data.

The tab must always be available, even if the independent RCM data loader fails.
No dataset of real service-life events is shipped with this release.
"""
from __future__ import annotations

import hashlib
from io import BytesIO

import altair as alt
import pandas as pd
import streamlit as st

from lifetime_ml import (
    LifetimeDataError, available_states, predict_lifetime, schema_csv_bytes,
    train_lifetime, REQUIRED_COLUMNS,
)
from rcm_training import AUTO, MODELS, predict_rcm


@st.cache_resource(show_spinner='Huấn luyện và kiểm định mô hình tuổi thọ bằng dữ liệu có nhãn...')
def _fit_lifespan(csv_content: bytes, limit_state: str, algorithm: str):
    raw = pd.read_csv(BytesIO(csv_content))
    return train_lifetime(raw, limit_state, algorithm)


def _input_range(name, interval, display_format='%.2f'):
    lo, hi = interval
    default = float((lo + hi) / 2.)
    return st.number_input(name, min_value=float(lo), max_value=float(hi), value=default,
                           format=display_format)


def render_lifespan_tab(rcm_result=None):
    st.subheader('Dự báo thời gian đạt trạng thái giới hạn bằng AI')
    st.info('**Chức năng v0.8 (không dùng Fick):** Mô hình chỉ được huấn luyện khi CSV có '
            'thời gian **đã quan sát** tới một trạng thái giới hạn xác định, được gắn nguồn '
            'nghiên cứu. Bảng RCM 215 mẫu KHÔNG có nhãn thời gian và KHÔNG thể dùng thay thế.')
    st.markdown('**1 · Nguồn dữ liệu thời gian thực nghiệm / theo dõi hiện trường**')
    a, b = st.columns([3, 2])
    with a:
        st.download_button('⬇ Tải CSV mẫu cấu trúc dữ liệu tuổi thọ (chưa có số liệu)',
                           data=schema_csv_bytes(),
                           file_name='CloriLife_TuoiThoAI_Mau_CauTruc_KhongDuLieu.csv',
                           mime='text/csv')
    with b:
        st.caption('Mẫu tải về **chỉ có tên cột**; tuyệt đối không chứa tuổi thọ giả định.')
    with st.expander('Giải thích các cột dữ liệu cần có', expanded=False):
        st.markdown('''
| Cột | Ý nghĩa |
|---|---|
| `fly_ash_pct`, `water_binder_ratio` | Hàm lượng tro bay (%) và tỷ lệ N/CKD |
| `d_rcm_28_m2_s` | Hệ số RCM thực nghiệm ở 28 ngày (m²/s), **không phải tuổi thọ** |
| `cover_mm` | Chiều dày lớp bê tông bảo vệ (mm) |
| `exposure_class` | Nhóm môi trường xác định thống nhất giữa các nguồn |
| `time_to_limit_years` | Thời gian quan sát **đến khi xảy ra trạng thái giới hạn** (năm) |
| `limit_state` | Định nghĩa cụ thể của trạng thái giới hạn, ví dụ `first_crack_0p3mm` |
| `event_observed` | `1` = trạng thái đã xảy ra; `0` = chưa xảy ra (censored) |
| `source_group`, `source_reference` | Mã nhóm nghiên cứu và tài liệu tham chiếu tra cứu được |

Tệp cần tối thiểu **40 bản ghi đủ nhãn và 8 nhóm nguồn độc lập** cho mỗi `limit_state`.

Không trộn dữ liệu *khởi phát ăn mòn*, *nứt lớp bảo vệ* và *suy giảm chịu lực* thành một nhãn tuổi thọ.

Với `event_observed=0`, cần xây dựng mô hình **survival analysis** riêng, không tự coi thời gian theo dõi là thời gian phá hoại.
''')
    uploaded = st.file_uploader('Tải dữ liệu tuổi thọ thật có nhãn (CSV UTF-8)',
                                type=['csv'], key='lifetime_observed_csv')
    if uploaded is None:
        st.warning('**Chưa có bộ dữ liệu tuổi thọ thực nghiệm.** Mô-đun đã sẵn sàng '
                   'nhận dữ liệu, nhưng hiện chưa thể đưa ra một số năm dự báo có cơ sở. '
                   '215 mẫu RCM chỉ phục vụ bài toán thuận/ngược của hệ số RCM.')
        st.caption('Để hoàn thiện mục tiêu đề tài, cần thu thập thời gian đến một trạng thái giới hạn '
                   'đã xảy ra và các biến cùng điều kiện môi trường. Không suy tuổi thọ từ RCM đơn thuần.')
        return

    content = uploaded.getvalue()
    try:
        raw = pd.read_csv(BytesIO(content))
        states = available_states(raw)
    except Exception as exc:
        st.error('Không đọc được cấu trúc dữ liệu thời gian: ' + str(exc))
        return
    state = st.selectbox('Chọn một trạng thái giới hạn để huấn luyện riêng',
                         states, key='life_limit_state')
    model_name = st.selectbox('Thuật toán dự báo thời gian', [AUTO] + list(MODELS),
                              key='life_algorithm',
                              help='Chọn tự động bằng MAE kiểm định chéo theo nguồn; '
                                   'không dùng holdout để chọn mô hình.')
    try:
        result = _fit_lifespan(content, state, model_name)
    except (LifetimeDataError, ValueError, KeyError) as exc:
        st.error('Không thể huấn luyện tuổi thọ: ' + str(exc))
        return
    except Exception as exc:
        st.error('Lỗi dữ liệu hoặc huấn luyện: ' + str(exc))
        return

    st.success(f'Đã kiểm định theo nguồn cho trạng thái: **{state}**. '
               f'Đang dùng mô hình: **{result["selected"]}**.')
    st.markdown('**2 · Kết quả kiểm định 9 thuật toán**')
    st.caption('Không sử dụng mẫu từ cùng nhóm nguồn ở cả train và holdout. '
               'Mô hình xếp hạng bằng kiểm định chéo trước khi kiểm tra holdout.')
    st.dataframe(result['table'].style.format({
        'CV MAE (năm)': '{:.2f}',
        'CV RMSE (năm)': '{:.2f}',
        'CV R²': '{:.3f}'
    }), hide_index=True, use_container_width=True)
    m = result['scores']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Nguồn nghiên cứu', result['train_sources'] + result['test_sources'])
    c2.metric('MAE holdout', f'{m["MAE_years"]:.2f} năm')
    c3.metric('RMSE holdout', f'{m["RMSE_years"]:.2f} năm')
    c4.metric('R² holdout', f'{m["R2"]:.3f}')
    st.caption(f'Train: {result["train_count"]} mẫu / {result["train_sources"]} nhóm nguồn. '
               f'Holdout: {result["test_count"]} mẫu / {result["test_sources"]} nhóm nguồn. '
               f'Baseline (trung vị train): MAE = {result["baseline_MAE_years"]:.2f} năm.')
    unreliable = m['R2'] <= 0 or m['MAE_years'] >= result['baseline_MAE_years']
    if unreliable:
        st.error('**Chưa đạt chất lượng dự báo đáng tin cậy:** Mô hình không vượt baseline '
                 'và/hoặc R² không dương trên nguồn giữ riêng. Tính năng dự báo tạm khóa để tránh suy diễn số năm tuổi thọ.')
    else:
        st.warning('Có kết quả kiểm tra bước đầu, nhưng độ tin cậy phụ thuộc nguồn dữ liệu, '
                   'định nghĩa trạng thái giới hạn và miền dự báo. Chưa được dùng để quyết định an toàn kết cấu.')
    hold = result['holdout']
    plot = alt.Chart(hold).mark_circle(size=75, opacity=.75).encode(
        x=alt.X('measured_time_to_limit_years:Q', title='Thời gian quan sát (năm)'),
        y=alt.Y('predicted_time_to_limit_years:Q', title='Thời gian AI dự báo (năm)'),
        tooltip=['source_group:N',
                 alt.Tooltip('measured_time_to_limit_years:Q', format='.2f'),
                 alt.Tooltip('predicted_time_to_limit_years:Q', format='.2f')]
    ).properties(height=280)
    st.altair_chart(plot, use_container_width=True)
    st.download_button('⬇ Tải bảng đánh giá trên nguồn giữ riêng',
                       data=hold.to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_AI_v08_KiemTra_TuoiTho.csv', mime='text/csv')

    st.markdown('**3 · Dự báo một kịch bản thuộc miền dữ liệu**')
    st.caption('Đầu ra là thời gian đến **đúng trạng thái giới hạn được ghi trong dữ liệu**; '
               'không tự động là tuổi thọ chịu lực của toàn bộ công trình.')
    with st.form('lifetime_prediction_form'):
        left, right = st.columns(2)
        with left:
            fly = _input_range('Tro bay (% chất kết dính)', result['bounds']['fly_ash_pct'])
            wb = _input_range('Tỷ lệ N/CKD', result['bounds']['water_binder_ratio'], '%.3f')
            cover = _input_range('Lớp bảo vệ cốt thép (mm)', result['bounds']['cover_mm'], '%.1f')
        with right:
            exposure = st.selectbox('Nhóm môi trường theo dữ liệu đã huấn luyện', result['exposures'])
            rcm_mode = st.radio('Nguồn hệ số RCM',
                                ['Nhập RCM đo thực tế'] +
                                (['Lấy RCM dự báo từ mô hình của phần mềm'] if rcm_result is not None else []),
                                help='RCM dự báo từ AI có sai số riêng, gây thêm bất định cho dự báo thời gian.')
            if rcm_mode == 'Nhập RCM đo thực tế':
                rcm_x1e12 = _input_range('RCM (×10⁻¹² m²/s)',
                                         result['bounds']['rcm_x1e12'], '%.3f')
                cement = grade = None
            else:
                cement = st.selectbox('Loại xi măng (mô hình RCM)', rcm_result['cement_types'])
                grade = st.selectbox('Cấp tro bay (mô hình RCM)', rcm_result['fly_ash_grades'])
                rcm_x1e12 = None
        go = st.form_submit_button('🔍 Dự báo thời gian đạt trạng thái giới hạn',
                                   type='primary', disabled=unreliable)
    key = hashlib.sha256(content).hexdigest() + '|' + state + '|' + model_name
    if go:
        try:
            if rcm_mode != 'Nhập RCM đo thực tế':
                rcm_x1e12 = predict_rcm(rcm_result, fly, wb, cement, grade) * 1e12
            predicted = predict_lifetime(result,
                                         fly_ash_pct=fly,
                                         water_binder_ratio=wb,
                                         d_rcm_28_m2_s=rcm_x1e12 * 1e-12,
                                         cover_mm=cover,
                                         exposure_class=exposure)
            st.session_state['life_prediction_last'] = {
                'key': key, 'estimated': predicted, 'rcm_x1e12': float(rcm_x1e12),
                'mode': rcm_mode, 'exposure': exposure, 'fly': fly, 'wb': wb, 'cover': cover,
            }
        except Exception as exc:
            st.session_state.pop('life_prediction_last', None)
            st.error('Không đưa ra số năm dự báo: ' + str(exc))
    previous = st.session_state.get('life_prediction_last')
    if not unreliable and previous and previous['key'] == key:
        st.metric(f'Thời gian dự báo đến giới hạn "{state}"',
                  f'{previous["estimated"]:.1f} năm')
        st.info(f'RCM đầu vào = {previous["rcm_x1e12"]:.3f} ×10⁻¹² m²/s '
                f'({previous["mode"]}). MAE trên holdout = {m["MAE_years"]:.2f} năm '
                '(chỉ số sai số tập kiểm tra, **không phải khoảng tin cậy 95%**).')
        st.warning('Đây là dự báo dựa trên dữ liệu có nhãn cùng một trạng thái giới hạn, '
                   'chưa phải kết luận tuổi thọ thiết kế. Các đặc trưng chưa đưa vào mô hình như '
                   'tải trọng, nứt, cường độ, nhiệt độ biến đổi có thể gây sai lệch đáng kể.')
        exported = pd.DataFrame([{ 'limit_state': state,
                                   'model': result['selected'],
                                   'fly_ash_pct': previous['fly'],
                                   'water_binder_ratio': previous['wb'],
                                   'cover_mm': previous['cover'],
                                   'exposure_class': previous['exposure'],
                                   'rcm_x1e12': previous['rcm_x1e12'],
                                   'rcm_source': previous['mode'],
                                   'predicted_time_to_limit_years': previous['estimated'],
                                   'holdout_mae_years': m['MAE_years'],
                                   'status': 'ML_ESTIMATE_NOT_DESIGN_VERIFIED' }])
        st.download_button('⬇ Tải dự báo tuổi thọ kèm trạng thái và giả định',
                           data=exported.to_csv(index=False).encode('utf-8-sig'),
                           file_name='CloriLife_AI_v08_DuBao_ThoiGian_Theo_GioiHan.csv',
                           mime='text/csv')
    st.caption('Phương pháp: hồi quy học máy, dùng log(1+t), chia nguồn GroupShuffleSplit '
               'và kiểm định GroupKFold. Không sử dụng định luật Fick.')
