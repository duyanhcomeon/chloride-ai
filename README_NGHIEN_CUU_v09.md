# CloriLife AI v0.9 — Thư viện nghiên cứu thực nghiệm ăn mòn cốt thép

*Cập nhật tài liệu và mã nguồn: 31/10/2026. Không sử dụng phương trình Fick trong phần mềm.*

## 1. Hai bài toán khác nhau

**Bài toán đã có:** Bộ dữ liệu `real_rcm_flyash_215.csv` gồm 215 dòng từ Excel Dryad, dùng so sánh 9 thuật toán dự báo hệ số di chuyển ion clorua RCM và tìm tổ hợp tro bay/N-CKD theo mục tiêu. RCM không phải tuổi thọ.

**Tài liệu nghiên cứu bổ sung:** 19 dòng ở `corrosion_initiation_19_groups.csv` thuộc **3 nghiên cứu độc lập**, gồm giá trị theo cấp phối và trung bình nhóm thí nghiệm. File `research_sources_14.csv` có **14 tài liệu** với URL đầy đủ. Phần mềm **chỉ hiển thị, lọc, xuất dữ liệu và truy xuất nguồn**; **chưa dùng 19 dòng này để huấn luyện hoặc cấp số năm tuổi thọ**.

| Nguồn | Số dòng | Đầu ra/điều kiện | Sử dụng phù hợp |
|---|---:|---|---|
| Menéndez, Argiz & Sanjuán (2019) | 10 | t_lag (giờ), thí nghiệm điện trường tăng tốc với vữa tro bay / tro đáy | So sánh ảnh hưởng cấp phối trong cùng giao thức |
| Idrees Zafar (2015) | 5 | Thời gian khởi phát (ngày) trung bình nhóm trong thử nghiệm NaCl | Phân tích dữ liệu có lặp nhưng chỉ trích trung bình nhóm |
| Park, Park & Jung (2020) | 4 | Thời gian khởi phát (ngày) dưới cấp dung dịch NaCl | So sánh theo loại chất kết dính trong cùng nghiên cứu |

**Không ghép trực tiếp** dữ liệu tăng tốc (giờ), ngâm muối (ngày), và phơi nhiễm tự nhiên (năm) để tạo nhãn chung `time_to_limit_years`.

## 2. Tại sao mô hình tuổi thọ chưa huấn luyện được từ 19 dòng?

- Chỉ có **3 nhóm nguồn nghiên cứu**, chưa đủ đánh giá độ chính xác trên những nghiên cứu độc lập. Tệp chỉ chứa các cấp phối/giá trị trung bình, không phải dữ liệu từng thanh thép.
- Thiếu nhiều cột cần để dùng `lifetime_ml.py`: `d_rcm_28_m2_s` cùng mẫu, điều kiện phơi nhiễm chuẩn hóa và đặc biệt **thời gian hư hỏng kết cấu chịu lực đã đo**.
- Kết quả thời gian của 19 dòng là **khởi phát ăn mòn trong các thí nghiệm cụ thể** chứ không phải tuổi thọ sử dụng của toàn kết cấu BTCT.
- Các nguồn có mẫu chưa xảy ra ăn mòn cần lưu **event_observed=0** và **censor_time**, không được xem chúng là mẫu “sống vô hạn”.

## 3. Cách xem trong app

1. Cập nhật tất cả file của `CloriLife_AI_v0_9_GitHub_Update.zip` vào **thư mục gốc** của dự án GitHub đang chạy v0.8, giữ nguyên các thư mục con.
2. Streamlit Cloud chạy `app.py`; nhấn tab **🔬 Thư viện nghiên cứu**.
3. Chọn 1 trong 3 nghiên cứu để xem biểu đồ; nút tải CSV có cả liên kết nguồn.
4. Mở từng nghiên cứu trong phần Danh mục để lấy URL DOI và trạng thái xác minh.
5. Tab **⏳ Tuổi thọ AI** vẫn cần một bộ CSV thời gian đến trạng thái giới hạn đủ mẫu và đúng cấu trúc; **không tự động lấy 19 dòng tổng hợp**.

## 4. Danh mục các tài liệu đã khảo sát (liên kết gốc)

| ID | Tác giả / năm | URL | Dạng bằng chứng |
|---|---|---|---|
| S01 | Menéndez, Argiz & Sanjuán (2019) | https://doi.org/10.3390/ma12121933 | 10 dòng t_lag tăng tốc đã trích |
| S02 | Idrees Zafar (2015) | https://doi.org/10.14943/doctoral.k12023 | 5 dòng trung bình t_init đã trích |
| S03 | Park, Park & Jung (2020) | https://doi.org/10.3390/ma13235430 | 4 dòng khởi phát đã trích |
| S04 | Lollini, Redaelli & Bertolini (2016) | https://doi.org/10.1617/s11527-015-0778-0 | 192 thanh, 15 thanh khởi phát trong ponding; có kiểm duyệt |
| S05 | Presuel-Moreno & Moreno (2016) | https://doi.org/10.14359/51689184 | Phơi nhiễm nước biển lâu dài, có tro bay |
| S06 | O’Reilly, Farshadfar & Darwin (2019) | https://doi.org/10.14359/51710968 | Ngưỡng clorua, thời điểm ăn mòn / tái thụ động |
| S07 | Gupta, Sugiyama & Islam (2019) | https://doi.org/10.11113/mjce.v31.16142 | Thời gian khởi phát theo tỷ số đối chứng |
| S08 | Martens et al. (2024) | https://doi.org/10.48804/WO62A4 | Bộ dữ liệu nứt do ăn mòn, không có nhãn tuổi thọ xác nhận |
| S09 | Mak, Desnerck & Lees (2019) | https://doi.org/10.17863/CAM.37247 | File XLSX nứt, mức ăn mòn, bám dính |
| S10 | Dahanni et al. (2026) | https://doi.org/10.57745/CMJUI3 | Hồ sơ clorua đến 36 năm, không có thời gian ăn mòn cho mỗi mẫu |
| S11 | Hoque & Presuel-Moreno (2025) | https://doi.org/10.31217/p.39.2.1 | Dòng ăn mòn theo thời gian (1600 ngày) |
| S12 | Novoa et al. (2018) | https://doi.org/10.17632/5ydtxnkn39.2 | Theo dõi điện hóa; cần kiểm tra lại toàn văn/metadata |
| S13 | Hartt, Nam & Li (2002) | https://trid.trb.org/View/731119 | Báo cáo FDOT; cần lấy bản gốc và trích bảng |
| S14 | Al-Amoudi và cộng sự | https://eprints.kfupm.edu.sa/1740/1/51-64_Durability_of_Reinforced_Concrete_in_Sulfate-Chloride_Sabkha_Media_and_Methods_of_Protection_Omar_Al-Amoudi.PDF | Môi trường sulfate-chloride; cần đối chiếu toàn văn |

### Nguồn dữ liệu RCM trước đó

Yang, Zhu & Wei, *Experimental data for chloride diffusion coefficient of concrete by rapid chloride migration test*, Dryad: https://doi.org/10.5061/dryad.66t1g1k6j

**Cảnh báo dữ liệu gốc:** Excel được gửi có 215 dòng Table 5, còn lịch sử Dryad ghi chú một số dòng trùng (209 và loại 13); vấn đề số lượng 215/196 vẫn chưa được đối chiếu đầy đủ. Không mặc định 215 mẫu là 215 mẫu hoàn toàn độc lập.

## 5. Nguyên tắc trích dẫn khi bảo vệ

- Khi nói “thời gian 455 giờ của cấp phối tro bay 25%”, chỉ rõ *Menéndez et al. (2019), Table 3*, thí nghiệm tăng tốc, không nói là 455 giờ ngoài môi trường thực.
- Khi nói “76 ngày” phải nói *Park et al. (2020), phần OCP monitoring*, chiều dày bảo vệ 20 mm, không đánh đồng kết quả dự báo tuổi thọ bằng Fick trong cùng bài báo với nhãn đo trực tiếp.
- Khi nói “192 thanh thép”, phải nói đây là **thiết kế thí nghiệm gồm thanh thép** và **15 thanh ghi nhận khởi phát trong ponding**, không phải 192 thời gian hư hỏng có thể dùng hồi quy.
- Khi phân tích các nghiên cứu khác nhau, chỉ so sánh phương pháp và độ phong phú dữ liệu, không suy trực tiếp rằng tro bay X% có thể kéo dài tuổi thọ kết cấu Y năm.

## 6. Hạn chế và hướng nâng cấp tiếp

Để có mô hình tuổi thọ kiểm chứng: xin bảng dữ liệu từng mẫu từ nhóm tác giả, chuẩn hóa nhãn khởi phát hoặc nứt/chịu lực thành các mô hình **riêng**; bổ sung `source_id`, `protocol`, `duration`, `event_observed`, `exposure`, `cover_mm`, `fly_ash_pct`, và các điều kiện vật liệu. Dùng **GroupKFold theo nguồn**, và khi có kiểm duyệt thì áp dụng **survival analysis**. Chỉ đánh giá khả năng ngoại suy ra công trình khi có số liệu thực địa và sai số độc lập.
