# CLORILIFE AI v0.9 - THUAN + NGUOC + TUOI THO AI CO NHAN THUC

# CloriLife AI v0.9 — học máy với bài toán thuận và bài toán ngược

**Nền tảng nghiên cứu bê tông tro bay: 215 mẫu RCM thực nghiệm, 9 thuật toán học máy. KHÔNG dùng định luật Fick.**

Nguồn gốc dữ liệu: Yang, Zhu và Wei, Dryad, DOI: https://doi.org/10.5061/dryad.66t1g1k6j. Bộ dữ liệu dùng ở đây là hệ số **chloride migration RCM ở 28 ngày**, không phải hệ số khuếch tán biểu kiến D_app.

## Cài đặt

Yêu cầu Python 3.12 hoặc 3.13, kết nối Internet khi cài gói:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Deploy: tải các tệp vào **thư mục gốc repository GitHub** của Streamlit Cloud, `app.py` là tệp khởi chạy. Gói ZIP cập nhật v0.8 dùng cho website đã chạy v0.6; gói toàn bộ mã nguồn dùng cho cài mới.

## Bài toán thuận

Nhập: tro bay (% khối lượng chất kết dính), tỷ lệ nước/chất kết dính N/CKD, loại xi măng, cấp tro bay.

Đầu ra: `D_RCM,28` dự báo (m²/s) từ mô hình học máy được chọn. Có 9 thuật toán: Linear Regression, Ridge, Decision Tree, Random Forest, Extra Trees, Gradient Boosting, SVR, KNN, MLP.

## Bài toán ngược (MỚI)

**Bài toán:** Cho trước ngưỡng mục tiêu `D_target`. Tìm một hoặc nhiều tổ hợp (tro bay %, N/CKD) trong một nhóm vật liệu cố định (loại xi măng + cấp tro bay), sao cho **RCM do mô hình dự báo không vượt ngưỡng**:

```
find  (FA_percent, W/B)
subject to:  model_predict_rcm(FA_percent, W/B, cement, grade) <= D_target
             low_FA <= FA_percent <= high_FA
             low_WB <= W/B <= high_WB
             (FA_percent, W/B) is in the measured training domain
```

Ba chế độ:

1. **Tìm đồng thời** tro bay và N/CKD.
2. **Tìm tro bay** khi giữ cố định N/CKD.
3. **Tìm N/CKD** khi giữ cố định hàm lượng tro bay.

Người dùng còn có thể chọn tiêu chí xếp hạng: gần mục tiêu, ít tro bay, N/CKD thấp, ưu tiên mẫu đã đo.

Thuật toán tìm kiếm **không cố suy ngược đại số** hoặc huấn luyện một mô hình giả tạo `RCM → cấp phối` vì bài toán **nhiều nghiệm**. Thay vào đó, sử dụng cùng mô hình thuận đã học, kiểm tra nhiều tổ hợp khả dĩ trong giới hạn dữ liệu:

- Chỉ dùng **176 dòng huấn luyện** (theo phép chia tập giữ lại hiện tại) làm điểm gốc để tìm nghiệm; 39 mẫu holdout không được dùng chọn ứng viên.
- Không ghép tùy ý loại xi măng và cấp tro bay chưa từng đi cùng nhau trong dữ liệu huấn luyện.
- Ứng viên mới phải nằm trong **bao lồi 2 chiều** của các mẫu thực nghiệm có cùng nhóm vật liệu và cách điểm gốc gần nhất tối đa 0,80 đơn vị khoảng cách chuẩn hóa.
- Ứng viên từ thí nghiệm gốc và ứng viên nội suy được đánh dấu **khác nhau**. Cột RCM đo của ứng viên nội suy để trống.
- Nếu không có ứng viên dự báo đạt mục tiêu, hiển thị rõ **KHÔNG TÌM ĐƯỢC NGHIỆM ĐẠT YÊU CẦU**, các phương án gần nhất không được ghi nhầm là đạt.
- Biên sàng lọc phụ là 0, 0,5 hoặc 1,0 lần **MAE kiểm định chéo trên tập huấn luyện**, không sử dụng MAE tập holdout để chọn thiết kế. Đây chỉ là khoảng đệm thực nghiệm, không là bảo đảm xác suất.
- Có bảng phương án, đồ thị và CSV truy xuất nguồn.

**Ví dụ thao tác:** Vào `🔁 Thiết kế ngược`, chọn PO425 / tro bay cấp 1.0, nhập RCM tối đa **8 × 10⁻¹² m²/s**, giữ khoảng tro bay và N/CKD theo dữ liệu, bấm `TÌM CÁC PHƯƠNG ÁN CẤP PHỐI` rồi so sánh các kết quả. Giá trị đầu ra thay đổi khi đổi mô hình hoặc ràng buộc.

## Dữ liệu, kiểm định và ranh giới khoa học

- Mô hình được chọn bằng GroupKFold theo nhóm công bố trên tập train. 20% mẫu giữ riêng theo nhóm bài báo để đánh giá tổng quát hóa.
- Dữ liệu gốc gồm 215 mẫu **đo RCM**, không bao gồm đầy đủ cường độ, độ sụt, thành phần cốt liệu, điều kiện dưỡng hộ cho phép thiết kế cấp phối hoàn chỉnh.
- Kết quả **chỉ là đề xuất tổ hợp hai thông số tro bay và N/CKD**, KHÔNG phải chỉ dẫn trộn bê tông đầy đủ, KHÔNG bảo đảm đạt chỉ tiêu RCM. Cần thí nghiệm RCM theo phương pháp tương thích, xác nhận chất lượng tro bay/xi măng và các yêu cầu kỹ thuật khác trước khi áp dụng.
- Dự báo RCM không tương đương dự báo thời gian khởi phát ăn mòn, tuổi thọ hoặc khả năng chịu lực công trình.

## Kiểm thử

```bash
python -m pip install pytest
python -m pytest -q tests/test_inverse_design.py tests/test_ml_only.py
```

Mã nền: `rcm_training.py`; mô-đun mới: `inverse_design.py`; giao diện: `app.py`. Không cần `physics.py`, `corrosion.py` hay các mô-đun Fick cũ.


## v0.8: scientific validation review (31 Oct 2026)

1. Measured rows are **screened by measured RCM**, never certified by an in-sample
   prediction. Proposed interpolation rows are screened **by predicted RCM only**,
   visibly marked as unverified, and must undergo RCM testing.
2. Contradictions between measured outcomes and model classifications are shown.
   Every scenario CSV contains the measured and predicted values, evidence level,
   and a disagreement flag. New proposal rows have blank experimental source fields;
   the nearest measured source is shown separately as *support*, not provenance.
3. No app-level `st.stop()` calls: when CSV/training fails, tabs remain visible and
   show explanations rather than causing the whole page to stop.
4. The supplied January-2025 XLSX has 215 data rows on Table 5, while Dryad's
   revision notes say Table 5 had 209 and then removed 13 duplicate records.
   This discrepancy remains **unresolved** and must be checked with the authors
   or an official version history before treating all rows as confirmed unique.
   The 215-row file is **not silently trimmed to 196 rows**.
5. On the currently defined 39-row independent group holdout, at the RCM target
   of 8 x 10^-12 m2/s, 18 rows are predicted to meet the target but 7 of those
   rows actually fail according to their reported measurements. This is a
   threshold-specific diagnostic, NOT a general failure rate.
6. `tests/` must be uploaded at the same folder level as app.py for tests to run.
   The GitHub UPDATE zip also carries the tests directory.

Release checked with 23 automated tests. Streamlit UI could not be executed
in this environment (Streamlit package unavailable); run it after deployment.


## Cap nhat v0.8: Tuoi tho AI KHONG FICK

- Co tab **Tuoi tho AI**, cho phep nap file CSV **co nhan thoi gian da quan sat** va nguon cua tung mau, hoc may 9 thuat toan theo GroupKFold.
- **Chua co du lieu tuoi tho that:** 215 mau Dryad chi co RCM, tab hien canh bao thay vi bịa ra con so nam.
- Can CSV rieng co `time_to_limit_years`, `limit_state`, `event_observed`, `source_group`, `source_reference` va cac thong so. File mau chi co ten cot.
- Neu `event_observed=0` (chua phat hien hong): dung lai, can survival analysis; khong coi tuoi quan sat la tuoi tho.
- Chi nhan nhan cung mot `limit_state`; mo hinh chon theo CV, kiem tra nguon doc lap o holdout.
- Xu ly de xuat thiet ke ngược RCM giu nguyen v0.7.1.
- Chi tiet xem `README_TUOI_THO_AI.md`.


## v0.9: Thu vien nghien cuu thoi gian khoi phat an mon (31/10/2026)

Them tab **🔬 Thư viện nghiên cứu**: 14 nguon tai lieu, 19 dong ket qua thoi gian khoi phat tu ba nghien cuu doc lap, loc va bieu do theo nguon, tai CSV.

- `research_sources_14.csv`: 14 URL DOI / duong dan bao cao, voi trang thai doi chieu.
- `corrosion_initiation_19_groups.csv`: 19 dong cap phoi/trung binh nhom, bao gom nhan, don vi goc, nguon va giao thuc.
- `research_catalog.py`: kiem tra nhan, so luong, tinh nhat quan cua quy doi gio/ngay va truy xuat nguon.
- `literature_ui.py`: giao dien khong tham gia huan luyen ML.
- `README_NGHIEN_CUU_v09.md`: giai thich day du va danh sach 14 link nghien cuu.

**Canh bao:** 19 dong khong phai 19 mau thu doc lap; chi co ba cong bo, nhieu giao thuc va thieu cac bien thoi gian tuoi tho ket cau. Tab Tuoi tho AI van yeu cau dataset rieng co nhan. KHONG dung Fick.

Kiem thu: `python -m pytest -q tests` (35 kiem thu)
