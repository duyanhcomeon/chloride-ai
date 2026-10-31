# CloriLife-AI v1.1 – Bắt đầu từ dữ liệu nào?

## 1. Chạy được ngay (dữ liệu RCM)
- Mở `CHAY_CLORILIFE_WINDOWS.bat` trên Windows (cần Python 3.12/3.13 + internet để cài các thư viện lần đầu).
- Hoặc CMD: `python -m pip install -r requirements.txt` rồi `python -m streamlit run app.py`.
- Trong sidebar chọn **Dryad — 215 mẫu tro bay**. KHÔNG phải tải tập tin nào trước.
- Xem tab **So sánh 9 mô hình**, **Dự báo thuận**, **Thiết kế ngược**. 215 dòng RCM phân theo 94 nhóm nguồn.
- Nếu muốn tự nạp CSV RCM: chọn **Tải CSV RCM của tôi**, dùng file `real_rcm_flyash_215.csv` làm ví dụ. Các cột: `fly_ash_pct,water_binder_ratio,cement_type,fly_ash_grade,d_rcm_28_m2_s,source_group`.
- 1268 dòng là bộ tổng hợp các loại vật liệu/RCM, không thể chuyển nguyên bộ vào mô hình 215 tro bay mà không làm lại chọn biến, chuẩn hóa phép thử và đánh giá theo nguồn. 557 dòng là tập con; không cộng 215+1268+557.

## 2. Tuổi thọ AI — chỉ huấn luyện khi có nhãn đúng
- Mở tab **Tuổi thọ AI**, chọn **Dự báo sống sót & bài toán ngược** và tải file CSV theo mẫu `CloriLife_v10_MAU_DU_LIEU_SURVIVAL.csv`.
- File mẫu **không chứa dữ liệu giả**: chỉ là cột tiêu đề. Phải thu thập từng mẫu, thời gian đến khi quan sát ăn mòn (hoặc thời điểm dừng quan sát), cờ `event_observed` (1/0), `source_group`, `limit_state`, `exposure_protocol`, `exposure_class`, `cover_mm`, tro bay %, N/CKD, DOI nguồn.
- Nếu kết quả chỉ xác định sự kiện xảy ra trong một khoảng giữa hai ngày đo, KHÔNG tự nhập ngày cuối như sự kiện chính xác; v1.1 chưa xử lý interval censoring.
- Phần mềm không ghép 215 RCM với 19 nhóm gia tốc để tự tạo nhãn tuổi thọ. 19 nhóm chỉ phục vụ đối chiếu tài liệu.
- Với dữ liệu có nhãn hợp lệ, mô-đun survival hiện có thể thử 3 thuật toán: Cox PH Ridge, Random Forest hazard theo thời gian, Gradient Boosting hazard theo thời gian. Các ngưỡng lọc là kiểm soát nội bộ, không chứng minh an toàn công trình.
- Bài toán ngược tuổi thọ chỉ được phép khi đã qua kiểm định nhóm nguồn và trong miền dữ liệu. Nếu không đạt, nút dự báo/thiết kế ngược bị khóa.

## 3. Tập tin cụ thể để người dùng nhận dạng
| File | Có thể dùng ngay để huấn luyện? | Đầu ra |
|---|---|---|
| real_rcm_flyash_215.csv | CÓ, dự báo RCM | Hệ số RCM 28 ngày |
| research_sources/RCM_8_TABLES_1268_RESEARCH.csv | CHƯA, phải chọn đồng nhất đầu vào | RCM nghiên cứu đa nhóm |
| corrosion_initiation_19_groups.csv | KHÔNG để huấn luyện tuổi thọ hiện trường | Thời gian quan sát thí nghiệm gia tốc |
| CloriLife_v10_MAU_DU_LIEU_SURVIVAL.csv | KHÔNG, chỉ header | Mẫu điền nhãn thời gian-sự kiện |
| research_sources/NEW_SOURCES_OCT_2026.csv | KHÔNG, chỉ đường dẫn nguồn | Khả năng thu thập số liệu mới |

## 4. Kiểm tra cấu trúc trước khi học
Chạy `python audit_inputs.py --type rcm --file real_rcm_flyash_215.csv` hoặc `python audit_inputs.py --type survival --file du_lieu_thoi_gian.csv`. Kết quả là báo cáo cơ bản giúp phát hiện thiếu cột, thiếu nguồn, cờ sự kiện sai. Công cụ kiểm tra cấu trúc không thay thế thẩm định dữ liệu gốc.

## 5. Nguồn
- Dryad Yang, Zhu & Wei (2025): https://doi.org/10.5061/dryad.66t1g1k6j
- Dahanni et al. (2026) chloride profiles đến 36 năm: https://doi.org/10.57745/CMJUI3
- Yilmaz et al. (2026) mô tả thời điểm khởi phát thí nghiệm: https://doi.org/10.1177/1478422X261419537
- Presuel-Moreno & Moreno (2016) phơi nhiễm biển: https://doi.org/10.14359/51689184

Kết luận: phần có thể báo cáo số liệu kiểm định độc lập lúc này là **RCM**, không phải số năm tuổi thọ.
