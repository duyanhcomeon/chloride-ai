# CloriLife AI v0.8 — Mô-đun thời gian đến trạng thái giới hạn bằng học máy

**KHÔNG dùng định luật Fick. KHÔNG tạo ra số năm tuổi thọ từ 215 mẫu RCM hiện có.**

## 1. Ba nhiệm vụ khác nhau

1. Bài toán thuận RCM: đầu vào cấp phối -> dự báo hệ số **D_RCM,28**.
2. Bài toán ngược RCM: đầu vào RCM mục tiêu -> đề xuất nhiều cấp phối, phân biệt mẫu đo thực và nội suy dự báo.
3. Tuổi thọ AI: đầu vào thông số vật liệu, RCM, lớp bảo vệ, môi trường -> dự báo **thời gian đến một trạng thái giới hạn đã quan sát**. Yêu cầu bộ dữ liệu thứ hai có nhãn thời gian, không gộp nhãn vật lý khác nhau.

## 2. CSV cho tab mới

Tải tệp `CloriLife_TuoiThoAI_CSV_Mau_CauTruc.csv`. File này CHỈ CÓ DÒNG TIÊU ĐỀ, không chứa số liệu giả.

| Cột | Bắt buộc | Nội dung |
|---|---|---|
| `fly_ash_pct` | Có | % tro bay theo khối lượng chất kết dính |
| `water_binder_ratio` | Có | Nước/chất kết dính |
| `d_rcm_28_m2_s` | Có | Hệ số RCM đo ở 28 ngày, đơn vị m²/s |
| `cover_mm` | Có | Bề dày bê tông bảo vệ, mm |
| `exposure_class` | Có | Nhóm môi trường, thống nhất trên các nguồn |
| `time_to_limit_years` | Có | Thời gian từ mốc ban đầu tới lúc **đã xảy ra** trạng thái giới hạn, năm |
| `event_observed` | Có | `1` nếu giới hạn đã xảy ra; `0` nếu chỉ quan sát đến thời điểm hiện tại |
| `source_group` | Có | Mã nghiên cứu/phòng thí nghiệm độc lập để chia tập theo nhóm nguồn |
| `limit_state` | Có | Mã sự kiện phải định nghĩa tường minh, ví dụ `first_crack_0p3mm` |
| `source_reference` | Có | DOI/URL/tài liệu nguồn cho từng mẫu |

**Cảnh báo:** Các sự kiện khác nhau như bắt đầu ăn mòn, nứt 0.3 mm, bong lớp bảo vệ và mất khả năng chịu lực **không phải một biến mục tiêu chung**. Phần mềm huấn luyện riêng theo `limit_state`.

Một công trình vẫn đang hoạt động khi kết thúc theo dõi có `event_observed=0` (kiểm duyệt phải). Không được dùng thời gian đã theo dõi như tuổi thọ thực. Bản v0.8 nhận biết và **từ chối** dữ liệu có censoring, vì các mô hình hồi quy thông thường không hỗ trợ dữ liệu này. Cần mô hình survival analysis chuyên biệt trong giai đoạn phát triển tiếp theo. Tham khảo: https://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

## 3. Kiểm định

- Ít nhất 40 quan sát có thời gian đến sự kiện đã đo, ít nhất 8 nhóm nguồn **cho mỗi loại sự kiện**.
- Mỗi nhóm nguồn chỉ ở một bên của tập chia train/holdout qua GroupShuffleSplit.
- Kiểm định chéo GroupKFold ở tập train để chọn giữa 9 thuật toán. Tập holdout chỉ đánh giá mô hình được chọn.
- MAE/RMSE (năm), R², baseline trung vị, biểu đồ thực đo / dự báo và file CSV kết quả.
- Tập holdout KHÔNG được dùng để chỉnh mô hình. Nếu thử nhiều cấu hình rồi dựa vào holdout để chọn, đánh giá không còn là kiểm tra độc lập.
- Dự báo chỉ cho phép các giá trị nằm **trong khoảng đã quan sát của tập huấn luyện** và môi trường đã gặp. Điều đó không bảo đảm an toàn nội suy đa chiều; vẫn phải kiểm tra mẫu gần nhất và nguồn áp dụng.
- Nếu RCM đầu vào được dự báo bằng AI thay vì đo, sẽ có thêm một lớp sai số và cần đánh giá lan truyền bất định trước khi dùng thực tế.

API: `clean_lifetime()`, `train_lifetime()`, `predict_lifetime()` trong `lifetime_ml.py`. Giao diện `lifespan_ui.py` được nhúng vào `app.py`.

**Không sử dụng kết quả dự báo như số năm tuổi thọ thiết kế, quyết định an toàn hay kiểm định chịu lực của công trình.** Mô hình sơ bộ chưa có biến tải trọng, tình trạng nứt, cơ học, điều kiện phơi nhiễm biến thiên theo thời gian.

## 4. Chạy

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
python -m pytest -q tests
```

Theo mặc định, tab **“Tuổi thọ AI”** hiển thị trạng thái *Chưa có dữ liệu tuổi thọ thực nghiệm*. Đây là hành vi ĐÚNG, không phải lỗi triển khai.
