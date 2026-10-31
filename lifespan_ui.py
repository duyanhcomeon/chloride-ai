"""CloriLife AI v1.0 survival analysis and experimental evidence dashboard."""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from survival_ml import (
    AUTO, MODELS, SurvivalDataError, available_cohorts, source_summary,
    schema_csv_bytes, train_survival, predict_survival, median_time, inverse_survival,
)
from lifespan_ui_legacy import render_lifespan_tab as render_legacy_lifespan_tab
from rcm_training import predict_rcm

ROOT = Path(__file__).resolve().parent


@st.cache_resource(show_spinner='Đang chia nguồn và kiểm định survival với mẫu kiểm duyệt...')
def _fit(csv_content: bytes, limit_state: str, protocol: str, selection: str):
    frame = pd.read_csv(BytesIO(csv_content), encoding='utf-8-sig')
    return train_survival(frame, limit_state, protocol, selection)


def _range_input(title: str, interval: tuple, fmt='%.3f', key=None):
    lower, upper = interval
    center = float((lower+upper)/2)
    return st.number_input(title, min_value=float(lower), max_value=float(upper),
                           value=center, format=fmt, key=key)


def _render_sources():
    st.markdown('#### Nguồn thí nghiệm trên thế giới — phân loại theo loại nhãn')
    df = pd.read_csv(ROOT / 'research_sources' / 'VERIFIED_STUDIES_2026.csv')
    c1, c2, c3 = st.columns(3)
    c1.metric('Nguồn tài liệu đã lập danh mục', len(df))
    c2.metric('Nguồn có dữ liệu thời gian hiện trường sẵn để huấn luyện',
              int(df['field_time_to_event_rows_ready'].eq('YES').sum()))
    c3.metric('Mẫu RCM đã có trong phần mềm', 215)
    st.warning('Có nghiên cứu dài hạn 15,6–17 năm và dữ liệu phơi nhiễm 36 năm, '
               'nhưng **chưa có bộ dữ liệu mẫu-đơn nhất đủ nhãn đầu vào + thời điểm ăn mòn**. '
               'Không được điền 15,6–17 năm cho mọi mẫu tro bay hoặc coi 36 năm phơi nhiễm '
               'là 36 năm tuổi thọ.')
    st.dataframe(df[['year', 'country', 'title', 'available_measurement',
                     'field_time_to_event_rows_ready', 'url']], hide_index=True,
                 use_container_width=True,
                 column_config={'url': st.column_config.LinkColumn('Nguồn gốc')})
    st.download_button('⬇ Tải danh mục DOI / nguồn thí nghiệm',
                       data=df.to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_v10_Nguon_Thuc_Nghiem.csv', mime='text/csv')

    st.markdown('#### Dữ liệu RCM quốc tế bổ sung (chưa có nhãn tuổi thọ)')
    more1, more2 = st.columns(2)
    with more1:
        st.download_button('⬇ Tải 1.268 dòng RCM từ 8 bảng',
            data=(ROOT/'research_sources'/'RCM_8_TABLES_1268_RESEARCH.csv').read_bytes(),
            file_name='CloriLife_RCM_8_bang_1268.csv', mime='text/csv')
    with more2:
        st.download_button('⬇ Tải 557 dòng RCM tro bay + xỉ',
            data=(ROOT/'research_sources'/'RCM_FLYASH_SLAG_557_RESEARCH.csv').read_bytes(),
            file_name='CloriLife_RCM_TroBay_Xi_557.csv', mime='text/csv')
    st.caption('Hai bảng là dữ liệu RCM đã sắp xếp theo nguồn từ v0.9. '
               'Không được cộng số mẫu các bảng để khẳng định có nghiên cứu độc lập, '
               'vì có thể có dòng trùng giữa các tập con. Chưa dùng để huấn luyện thời gian ăn mòn.')

    st.markdown('#### 19 bản ghi thời gian ăn mòn thí nghiệm hiện có')
    old = pd.read_csv(ROOT / 'corrosion_initiation_19_groups.csv')
    cols = st.columns(3)
    cols[0].metric('Bản ghi điều kiện thí nghiệm', len(old))
    cols[1].metric('Nguồn nghiên cứu', old['study_id'].nunique())
    cols[2].metric('RCM 28 ngày đo cùng mẫu', 'Không có')
    st.caption('Bảng gốc dùng thời gian **giờ/ngày trong điều kiện gia tốc**; '
               'không thể gộp trực tiếp thành năm tuổi thọ ngoài hiện trường. '
               'Một số kết quả là trung bình theo cấp phối, không phải các mẫu độc lập.')
    st.dataframe(old[['study_id','mix_id','fly_ash_pct','water_binder_ratio',
                      'cover_mm','time_to_corrosion_days','condition','source_url']],
                 hide_index=True, use_container_width=True,
                 column_config={'source_url': st.column_config.LinkColumn('Tài liệu')})
    st.download_button('⬇ Tải 19 bản ghi gốc (không chuyển thành năm hiện trường)',
                       data=old.to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_19_BanGhi_ThiNghiem_GiaToc.csv', mime='text/csv')
    st.info('Ưu tiên tiếp theo: xin bảng kết quả theo từng mẫu của ACI 2016; '
            'thu thập trạng thái ăn mòn và thời gian kiểm tra từ Thụy Điển; '
            'đối chiếu dữ liệu chloride profiles ở Pháp với công bố thời điểm khởi phát. '
            'Chỉ nhập dữ liệu đã được xác minh theo dòng và nguồn.')


def _render_training(rcm_result):
    st.markdown('#### AI dự báo thời gian đến khi bắt đầu ăn mòn — xử lý mẫu chưa xảy ra sự kiện')
    st.info('**Mô hình survival v1.0, không dùng Fick.** Mẫu chưa ăn mòn (`event_observed = 0`) '
            'được xem là **kiểm duyệt phải**, không bị coi là đã hư hỏng. '
            'Cox PH + Random Forest Survival + Gradient Boosting Survival '
            'được kiểm định chéo theo **nhóm nguồn nghiên cứu**.')
    st.download_button('⬇ Tải CSV mẫu tuổi thọ v1.0 (chỉ tiêu đề, KHÔNG có dữ liệu giả)',
                       data=schema_csv_bytes(),
                       file_name='CloriLife_v10_Mau_Co_Censoring.csv', mime='text/csv')
    with st.expander('Giải thích dữ liệu, nguồn và quy tắc nghiên cứu', expanded=False):
        st.markdown('''
- `observed_duration_years`: thời gian quan sát đến sự kiện nếu `event_observed=1`, hoặc thời gian **theo dõi cuối cùng** nếu `event_observed=0`.
- `limit_state`: chỉ được huấn luyện **một loại trạng thái giới hạn** mỗi lần (ví dụ `corrosion_initiation`, không trộn với `first_crack_0p3mm`).
- `exposure_protocol`: **một điều kiện thí nghiệm/hiện trường** mỗi lần (`natural_marine`, `accelerated_electrical`...). Không coi thí nghiệm gia tốc là năm tuổi thọ tự nhiên.
- `source_group`, `source_reference`: mã nhóm phòng thí nghiệm/công bố và URL/DOI xuất xứ. Tối thiểu 8 nhóm **độc lập**, 40 mẫu, 12 sự kiện; đây là ngưỡng kỹ thuật tối thiểu, **không tự khẳng định đủ độ tin cậy**.
- `fly_ash_pct`, `water_binder_ratio`, `cover_mm`, `exposure_class`: biến đầu vào bắt buộc.
- `d_rcm_28_m2_s`: **không bắt buộc**, nếu không đo đầy đủ cùng bộ tuổi thọ sẽ loại RCM khỏi mô hình survival; tuyệt đối không điền RCM giả để giữ số cột.

Huấn luyện theo nguồn với `GroupKFold` và holdout theo nguồn. Với thí nghiệm không có thời điểm ăn mòn chính xác (chỉ biết ăn mòn giữa hai lần kiểm tra), cần phương pháp kiểm duyệt khoảng riêng; không tùy tiện điền thời điểm cuối đợt đo. Dự báo tại thời gian ngoài miền theo dõi bị chặn.
''')
    uploaded = st.file_uploader('Tải CSV thí nghiệm có nhãn thời gian thực tế',
                                type=['csv'], key='survival_v10_data')
    if uploaded is None:
        st.warning('**Chưa đủ dữ liệu để công bố tuổi thọ theo năm.** '
                   'Phần mềm đang có 215 mẫu RCM và 19 bản ghi thí nghiệm ăn mòn gia tốc; '
                   'các dữ liệu này không được ghép thành nhãn tuổi thọ tự nhiên. '
                   'Hãy dùng mục “Nguồn thí nghiệm” để xem bộ tài liệu và tiến độ xác minh.')
        return
    content = uploaded.getvalue()
    try:
        df = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
        summary = source_summary(df)
        cohorts = available_cohorts(df)
        if not cohorts:
            raise SurvivalDataError('Chưa có nhóm trạng thái giới hạn/phơi nhiễm có dữ liệu.')
    except Exception as ex:
        st.error('Không đọc được CSV nghiên cứu: ' + str(ex))
        return
    st.markdown('**Bước 1 — Kiểm tra nguồn và trạng thái giới hạn**')
    st.dataframe(summary, hide_index=True, use_container_width=True)
    cohort = st.selectbox('Nhóm dữ liệu độc lập cần huấn luyện', cohorts,
                          format_func=lambda x: f'{x[0]} / {x[1]}', key='survival_cohort')
    chosen = st.selectbox('Lựa chọn thuật toán sống sót', [AUTO, *MODELS],
                          help='Chọn tự động bằng C-index kiểm định chéo theo nguồn. '
                               'Tập holdout không được dùng để chọn thuật toán.',
                          key='survival_algorithm')
    try:
        result = _fit(content, cohort[0], cohort[1], chosen)
    except (SurvivalDataError, ValueError, KeyError) as ex:
        st.error('Chưa thể huấn luyện và kiểm định: ' + str(ex))
        return
    except Exception as ex:
        st.error('Lỗi xử lý dữ liệu: ' + str(ex))
        return
    st.markdown('**Bước 2 — Kết quả kiểm định độc lập theo nguồn**')
    e = result['evaluation']
    top = st.columns(4)
    top[0].metric('Mẫu đã dùng', result['source_data_count'])
    top[1].metric('Mẫu chưa xảy ra ăn mòn', result['censored_count'])
    top[2].metric('C-index holdout', f'{e["C_index"]:.3f}')
    top[3].metric('IPCW Brier holdout', f'{e["IPCW_Brier"]:.3f}')
    st.caption(f'Mô hình: **{result["selected"]}**. Train {result["train_count"]} mẫu / '
               f'{result["train_sources"]} nguồn; holdout {result["test_count"]} mẫu / '
               f'{result["test_sources"]} nguồn. Các nguồn train/holdout không trùng. '
               f'C-index dựa trên {e["comparable_pairs"]} cặp so sánh hợp lệ.')
    st.dataframe(result['table'].style.format({'CV C-index': '{:.3f}',
                                                'CV IPCW Brier':'{:.3f}'}),
                 hide_index=True, use_container_width=True)
    st.caption(f'Mốc đánh giá: {result["horizon"]:.2f} năm theo dõi. '
               f'IPCW Brier baseline Kaplan–Meier (train): {e["baseline_IPCW_Brier"]:.3f}. '
               'IPCW giả định cơ chế kiểm duyệt không phụ thuộc sự kiện khi đã điều kiện hóa phù hợp; '
               'điểm số của nhóm nguồn nhỏ có thể rất bất ổn.')
    st.download_button('⬇ Tải kết quả holdout theo nguồn',
                       data=result['holdout'].to_csv(index=False).encode('utf-8-sig'),
                       file_name='CloriLife_v10_Holdout_Survival.csv', mime='text/csv')
    if not result['pass_quality']:
        st.error('**Khóa dự báo số năm và bài toán ngược.** Chất lượng holdout chưa vượt '
                 'baseline và điều kiện C-index / số cặp so sánh. Có thể xem báo cáo '
                 'nghiên cứu, nhưng cần bổ sung nguồn và kiểm chứng trước khi dùng kịch bản.')
        return
    st.success('Kiểm định holdout đạt ngưỡng sơ bộ của phần mềm. Chỉ cho phép '
               '**dự báo tham khảo trong miền nghiên cứu**, không phải tuổi thọ thiết kế.')
    if result['model_uses_rcm']:
        st.caption('RCM 28 ngày đo thực nghiệm đầy đủ đã được dùng làm đặc trưng. '
                   'Nếu dùng RCM dự báo bằng AI sẽ phát sinh thêm sai số chưa được định lượng.')
    else:
        st.warning('Không đủ RCM đo ghép theo từng mẫu, nên mô hình tuổi thọ '
                   '**không** sử dụng RCM làm biến đầu vào.')

    st.markdown('**Bước 3 — Dự báo kịch bản trong miền quan sát**')
    with st.form('survival_scenario_v10'):
        a, b = st.columns(2)
        with a:
            fa = _range_input('Tro bay (% chất kết dính)', result['bounds']['fly_ash_pct'])
            wb = _range_input('N/CKD', result['bounds']['water_binder_ratio'])
            cover = _range_input('Lớp bảo vệ (mm)', result['bounds']['cover_mm'], fmt='%.1f')
        with b:
            env = st.selectbox('Môi trường trong dữ liệu', result['exposures'])
            yrmax = result['max_years']
            if yrmax <= 1:
                target_days = st.number_input('Mốc thời gian thí nghiệm (ngày)',
                                              min_value=.01, max_value=float(yrmax*365.25),
                                              value=float(min(yrmax*365.25, 30.)), format='%.2f')
                target = target_days / 365.25
            else:
                target = st.number_input('Mốc nghiên cứu (năm)', min_value=.01,
                                         max_value=float(yrmax),
                                         value=float(min(result['horizon'], yrmax)), format='%.2f')
            if result['model_uses_rcm']:
                rcm_modes = ['RCM đo thực nghiệm'] + (
                    ['RCM dự báo bởi AI (sai số cộng dồn)'] if rcm_result is not None else [])
                rcm_mode = st.radio('Nguồn hệ số RCM', rcm_modes, horizontal=True)
                if rcm_mode == 'RCM đo thực nghiệm':
                    rx = _range_input('RCM ×10⁻¹² m²/s', result['bounds']['rcm_x1e12'])
                    cement = grade = None
                else:
                    cement = st.selectbox('Xi măng cho RCM AI', rcm_result['cement_types'])
                    grade = st.selectbox('Cấp tro bay cho RCM AI', rcm_result['fly_ash_grades'])
                    rx = None
            else:
                rcm_mode = 'Không dùng RCM'
                rx = cement = grade = None
        submit = st.form_submit_button('🔎 Dự báo theo dữ liệu thực nghiệm', type='primary')

    if submit:
        try:
            frame = pd.DataFrame([dict(fly_ash_pct=fa, water_binder_ratio=wb,
                                       cover_mm=cover, exposure_class=env)])
            if result['model_uses_rcm']:
                if rx is None:
                    rx = predict_rcm(rcm_result, fa, wb, cement, grade)*1e12
                frame['rcm_x1e12'] = float(rx)
            years = np.linspace(max(.00001, target/30), result['max_years'], 85)
            pred = predict_survival(result, frame, years)
            p_target = predict_survival(result, frame, [target]).iloc[0]
            med = median_time(result, frame)[0]
            st.session_state['clori_v10_last'] = dict(
                signature=(hash(content), cohort, chosen),
                p_target=float(p_target['survival_probability']),
                target=float(target), median=med, curve=pred,
                row=frame, rcm_mode=rcm_mode)
        except (SurvivalDataError, ValueError) as exc:
            st.session_state.pop('clori_v10_last', None)
            st.error(str(exc))
    previous = st.session_state.get('clori_v10_last')
    if previous and previous['signature'] == (hash(content), cohort, chosen):
        st.markdown('**Kết quả dự báo theo mô hình đã kiểm định**')
        q1, q2, q3 = st.columns(3)
        q1.metric('Xác suất chưa xuất hiện ăn mòn tại mốc',
                  f'{previous["p_target"]*100:.1f}%')
        if previous['median'] is None:
            q2.metric('Thời gian trung vị dự báo', 'Chưa xác định')
        else:
            q2.metric('Thời gian trung vị theo dữ liệu',
                      f'{previous["median"]*365.25:.1f} ngày' if result['max_years'] <= 1
                      else f'{previous["median"]:.1f} năm')
        q3.metric('Mốc đang xét',
                  f'{previous["target"]*365.25:.1f} ngày' if result['max_years'] <= 1
                  else f'{previous["target"]:.1f} năm')
        chart = previous['curve'].copy()
        chart['time_display'] = (chart['year']*365.25 if result['max_years'] <= 1
                                 else chart['year'])
        xaxis = ('Thời gian thí nghiệm (ngày)' if result['max_years'] <= 1
                 else 'Thời gian theo dõi (năm)')
        figure = alt.Chart(chart).mark_line(strokeWidth=3).encode(
            x=alt.X('time_display:Q', title=xaxis),
            y=alt.Y('survival_probability:Q', title='P(chưa xảy ra sự kiện)', scale=alt.Scale(domain=[0,1])),
            tooltip=[alt.Tooltip('time_display:Q', format='.2f'),
                     alt.Tooltip('survival_probability:Q', format='.3f')]
        ).properties(height=280)
        st.altair_chart(figure, use_container_width=True)
        st.warning('Cảnh báo: đây là **xác suất có điều kiện từ dữ liệu nghiên cứu**, '
                   'không phải xác suất an toàn của kết cấu hoặc khoảng tin cậy. '
                   'Mô hình không chứa thông tin tải trọng, vết nứt và biến thiên môi trường '
                   'đầy đủ. Nếu dùng RCM dự báo bằng AI, chưa có đánh giá lan truyền sai số.')
        if 'accelerated' in result['exposure_protocol'].lower():
            st.error('DỮ LIỆU GIA TỐC: thời gian hiển thị là thời gian thí nghiệm, '
                     'không được diễn giải là tuổi thọ ngoài hiện trường.')
        export = previous['curve'].copy()
        export['limit_state'] = result['limit_state']
        export['exposure_protocol'] = result['exposure_protocol']
        export['model'] = result['selected']
        export['C_index_holdout'] = result['evaluation']['C_index']
        export['report_status'] = 'RESEARCH_ONLY_NOT_DESIGN_LIFE'
        st.download_button('⬇ Tải đường cong dự báo theo kịch bản',
                           data=export.to_csv(index=False).encode('utf-8-sig'),
                           file_name='CloriLife_v10_Survival_Curve.csv', mime='text/csv')

    st.markdown('**Bước 4 — Bài toán ngược: tìm cấp phối theo mốc nghiên cứu**')
    st.caption('Tối ưu xác suất chưa đạt trạng thái giới hạn tại mốc đã chọn; '
               'chỉ quét trong miền mẫu huấn luyện. Không tự khẳng định cấp phối đạt tuổi thọ thiết kế.')
    with st.form('survival_inverse_v10'):
        c1, c2 = st.columns(2)
        with c1:
            year = st.number_input('Mốc cần đạt (năm thời gian nghiên cứu)',
                                   min_value=.001, max_value=float(result['max_years']),
                                   value=float(min(result['horizon'], result['max_years'])),
                                   format='%.3f', key='survival_inverse_horizon')
            minprob = st.number_input('Xác suất chưa ăn mòn mục tiêu (%)', min_value=1.,
                                      max_value=99., value=80., step=1.)
        with c2:
            cover_inv = _range_input('Lớp bê tông bảo vệ (mm) cho bài toán ngược',
                                     result['bounds']['cover_mm'], fmt='%.1f', key='cover_inv')
            env_inv = st.selectbox('Nhóm môi trường cho bài toán ngược',
                                   result['exposures'], key='env_inv')
            if result['model_uses_rcm'] and rcm_result is not None:
                cement_inv = st.selectbox('Xi măng (RCM AI)', rcm_result['cement_types'], key='cement_inv')
                grade_inv = st.selectbox('Cấp tro bay (RCM AI)', rcm_result['fly_ash_grades'], key='grade_inv')
            else:
                cement_inv = grade_inv = None
        inverse_submit = st.form_submit_button('🔁 Tìm cấp phối thử nghiệm')
    if inverse_submit:
        try:
            provider = None
            if result['model_uses_rcm'] and rcm_result is not None:
                provider = lambda f, w: predict_rcm(rcm_result, f, w, cement_inv, grade_inv)
            suggestions = inverse_survival(result, target_years=year,
                                           minimum_survival=minprob/100,
                                           cover_mm=cover_inv,
                                           exposure_class=env_inv,
                                           rcm_predictor=provider)
            if not suggestions['dat_nguong_xac_suat'].any():
                st.warning('Không tìm được phương án đạt xác suất mục tiêu trong '
                           'miền dữ liệu. Bảng dưới là các phương án gần mục tiêu nhất.')
            st.dataframe(suggestions, hide_index=True, use_container_width=True)
            st.download_button('⬇ Tải danh sách cấp phối thử nghiệm',
                               data=suggestions.to_csv(index=False).encode('utf-8-sig'),
                               file_name='CloriLife_v10_BaiToanNguoc_TuoiTho.csv', mime='text/csv')
        except (SurvivalDataError, ValueError) as exc:
            st.error('Chưa đưa ra được cấp phối: ' + str(exc))


def render_lifespan_tab(rcm_result=None):
    st.subheader('CloriLife AI v1.0 | Tuổi thọ từ dữ liệu thực nghiệm')
    tabs = st.tabs(['Dự báo sống sót & bài toán ngược',
                    'Nguồn thí nghiệm quốc tế',
                    'Hồi quy thời gian đủ nhãn (v0.9)'])
    with tabs[0]:
        _render_training(rcm_result)
    with tabs[1]:
        _render_sources()
    with tabs[2]:
        st.caption('Bộ hồi quy 9 mô hình cũ chỉ phù hợp khi toàn bộ thời điểm '
                   'sự kiện đã được quan sát và dữ liệu có nguồn độc lập. '
                   'Để dùng dữ liệu có censoring, chuyển sang tab đầu.')
        render_legacy_lifespan_tab(rcm_result)
