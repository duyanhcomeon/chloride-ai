import io
import numpy as np
import pandas as pd
import streamlit as st
from physics import concentration, initiation_years
from ml import FEATURES, fit_model, predict

st.set_page_config(page_title='Chloride-AI | BTCT tro bay', layout='wide', page_icon='🏗️')
st.title('🏗️ Chloride-AI | Tuổi khởi phát ăn mòn BTCT')
st.caption('Phiên bản MVP v0.1 · mô hình lai dữ liệu + Fick · dùng cho nghiên cứu và sàng lọc, KHÔNG dùng để quyết định thiết kế.')
st.warning('Kết quả là **thời gian khởi phát ăn mòn do clorua**, KHÔNG phải tuổi thọ kết cấu. Chưa mô phỏng nứt, suy giảm tiết diện thép, độ tin cậy hoặc khả năng chịu lực.')

with st.sidebar:
    st.header('1. Hệ số khuếch tán')
    mode = st.radio('Nguồn D28', ['Nhập hệ số đã đo', 'AI huấn luyện từ CSV'], index=0)
    model=None; bounds=None
    if mode == 'AI huấn luyện từ CSV':
        uploaded=st.file_uploader('CSV kết quả thử nghiệm (đúng các cột hướng dẫn)',type=['csv'])
        if uploaded is not None:
            try:
                df=pd.read_csv(uploaded)
                model, metrics, bounds=fit_model(df)
                st.success(f"Huấn luyện xong: {metrics['n']} mẫu | MAE = {metrics['MAE_m2_s']:.2e} m²/s | R² log = {metrics['R2_log10']:.3f}")
                st.caption('Random split chỉ kiểm tra nội bộ, không chứng minh ngoại suy cho cấp phối/công trình mới.')
            except Exception as ex:
                st.error(str(ex))
    st.header('2. Vật liệu')
    fly_ash=st.number_input('Tro bay / chất kết dính (%)',0.,80.,25.,1.)
    wb=st.number_input('Nước / chất kết dính',0.15,0.90,0.40,0.01)
    curing=st.number_input('Tuổi mẫu khi thử (ngày)',1,3650,28)
    strength=st.number_input('Cường độ nén (MPa)',5.,150.,40.,1.)
    temp=st.number_input('Nhiệt độ tham chiếu (°C)',0.,60.,25.,1.)
    if mode == 'AI huấn luyện từ CSV' and model is not None:
        vals=dict(zip(FEATURES,[fly_ash,wb,curing,strength,temp]))
        d28, outside=predict(model,vals,bounds)
        st.metric('D28 dự báo (m²/s)',f'{d28:.3e}')
        if outside: st.error('Đầu vào ngoài miền dữ liệu huấn luyện: '+', '.join(outside))
        st.info('Dữ liệu huấn luyện PHẢI có nhãn là D ở tuổi tham chiếu 28 ngày, đồng nhất phương pháp đo và đơn vị.')
    else:
        d28=st.number_input('D28 đo được (×10⁻¹² m²/s)', min_value=0.01,max_value=100.,value=8.,step=.5)*1e-12
    m=st.number_input('Số mũ già hóa m (giả định/hiệu chuẩn)',0.00,0.95,0.35,0.05)

st.subheader('3. Kết cấu và môi trường')
a,b,c=st.columns(3)
with a:
    cover=st.number_input('Lớp bảo vệ cốt thép (mm)',min_value=5.,max_value=200.,value=50.)
    horizon=st.slider('Thời gian phân tích tối đa (năm)',10,200,100)
with b:
    cs=st.number_input('Clorua bề mặt Cs (% khối lượng chất kết dính)',min_value=0.01,max_value=10.,value=1.2,step=.1)
    c0=st.number_input('Clorua ban đầu C0 (cùng đơn vị Cs)',min_value=0.,max_value=10.,value=0.,step=.05)
with c:
    critical=st.number_input('Ngưỡng Ccrit (cùng đơn vị Cs)',min_value=0.01,max_value=10.,value=0.4,step=.05)
    st.caption('Cs, C0, Ccrit phải cùng cơ sở khối lượng. Ngưỡng là giả định của người dùng.')

if not (c0 < critical < cs):
    st.error('Cần C0 < Ccrit < Cs. Chỉnh các giá trị trước khi tính.')
else:
    try:
        years=initiation_years(cover,d28,m,cs,critical,c0,horizon)
        if years is None:
            st.info(f'Chưa chạm ngưỡng trong {horizon} năm theo giả định mô hình. KHÔNG được kết luận tuổi thọ > {horizon} năm.')
        else:
            st.metric('Thời gian dự đoán đến ngưỡng Cl⁻ tại thép',f'{years:.1f} năm')
        times=np.linspace(0.001,horizon,300)
        vals=concentration(times,cover,d28,m,cs,c0)
        plot=pd.DataFrame({'Năm':times,'Cl⁻ tại thép (% khối lượng CKD)':vals,'Ngưỡng Ccrit':np.full_like(times,critical)}).set_index('Năm')
        st.line_chart(plot)
        result=pd.DataFrame({'time_years':times,'chloride_pct_binder':vals,'threshold_pct_binder':critical})
        st.download_button('Tải CSV đường cong dự báo',result.to_csv(index=False).encode('utf-8-sig'),file_name='chloride_prediction.csv',mime='text/csv')
        st.dataframe(pd.DataFrame([{'D28_m2_s':d28,'m':m,'cover_mm':cover,'Cs_pct_binder':cs,'C0_pct_binder':c0,'Ccrit_pct_binder':critical,'horizon_years':horizon,'initiation_years':years,'model_limit':'screening only'}]),hide_index=True)
    except ValueError as ex: st.error(str(ex))

with st.expander('Phương pháp, yêu cầu dữ liệu và giới hạn'):
    st.markdown('''**Vật lý:** Với D(t) chỉ phụ thuộc thời gian, nghiệm bán vô hạn dùng thời gian khuếch tán tích lũy I(t)=∫D(t)dt; C(x,t)=C0+(Cs−C0)erfc[x/(2√I(t))]. D(t)=D28 trước 28 ngày và D28×(28 ngày/t)^m sau 28 ngày. Điều kiện Cs không đổi, môi trường đồng nhất, không nứt, vận chuyển 1 chiều; bỏ qua trường hợp ướt/khô tuần hoàn và các cơ chế kết hợp.

**AI:** Đọc CSV với các cột `fly_ash_pct,water_binder_ratio,curing_days,compressive_strength_mpa,temperature_c,d28_m2_s`. Nhãn `d28_m2_s` phải cùng phương pháp thí nghiệm, quy đổi hợp lệ về 28 ngày, và bao gồm thông tin chất lượng; chỉ dùng random train/test split minh họa. Không chuyển đổi trực tiếp chỉ số RCPT thành D. Nếu chưa có dữ liệu thực nghiệm, hãy dùng chế độ nhập D đo được.

**Giới hạn quan trọng:** Mặc định là số giả định, không phải thông số đã kiểm định. Mô hình không xử lý ràng buộc ion, độ ẩm, nhiệt độ biến đổi, nứt, carbonat hóa, độ không chắc chắn, giai đoạn lan truyền ăn mòn. Phải hiệu chuẩn bằng thí nghiệm và quan trắc tại công trình trước khi áp dụng thực tế.''')
