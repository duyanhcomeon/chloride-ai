# CloriLife-AI v1.0 | RCM + Survival AI + Thiết kế ngược

Phiên bản cập nhật ngày **31/10/2026**, phát triển tiếp từ CloriLife-AI v0.9. Phần mềm viết bằng Python và Streamlit, chạy trên máy tính hoặc triển khai lên GitHub/Streamlit Community Cloud.

**Không sử dụng định luật Fick để suy số năm tuổi thọ. Không tạo dữ liệu tuổi thọ giả từ bộ RCM.**

## Chức năng thực sự

| Chức năng | Đầu vào | Đầu ra | Tình trạng dữ liệu |
|---|---|---|---|
| Dự báo RCM | Tro bay %, N/CKD, loại xi măng, cấp tro bay | RCM ×10⁻¹² m²/s, so sánh **9 thuật toán**, holdout theo nhóm nguồn | 215 bản ghi RCM từ dữ liệu công bố; cần làm rõ chênh lệch phiên bản Dryad |
| Bài toán ngược RCM | RCM mục tiêu và miền cấp phối | Tập ứng viên trong miền đã đo; đánh dấu thực nghiệm/ứng viên AI | Hoạt động với 215 mẫu RCM hiện có; phải thử nghiệm kiểm chứng |
| **Survival AI** | CSV mẫu có thời gian đến một trạng thái giới hạn + kiểm duyệt phải | Xác suất chưa xảy ra sự kiện tại mốc thời gian, đồ thị S(t), thời gian trung vị nếu xác định được | **Chưa có dữ liệu thực nghiệm tự nhiên đủ điều kiện để huấn luyện ngay** |
| **Bài toán ngược thời gian** | Mốc năm và xác suất yêu cầu, lớp bảo vệ, môi trường | Gợi ý tổ hợp cấp phối trong miền dữ liệu sau khi holdout đạt ngưỡng | Khóa cho tới khi có bộ dữ liệu được kiểm chứng |
| **Danh mục thí nghiệm quốc tế** | DOI, báo cáo, bảng nghiên cứu | Danh sách 12 nguồn và 19 bản ghi gia tốc có thể truy xuất | Có thật nhưng chưa tự động tương đương số năm tuổi thọ |

### Cài và chạy trên Windows

1. Giải nén toàn bộ ZIP vào một thư mục, ví dụ `D:\CloriLife_AI_v1_0`.
2. Cài **Python 3.12 hoặc 3.13** từ python.org; tích chọn `Add Python to PATH`.
3. Nhấn đúp `CHAY_CLORILIFE_WINDOWS.bat` để cài thư viện và mở Streamlit.
4. Nếu không mở bằng BAT, mở CMD/PowerShell trong thư mục vừa giải nén và chạy:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Trình duyệt mở `http://localhost:8501` (hoặc cổng khác do Streamlit thông báo).

### Deploy Streamlit Cloud

1. Giải nén và đưa **toàn bộ file trong thư mục** này lên thư mục gốc của một repository GitHub. Không đưa riêng file `app.py` vì còn các module và dữ liệu CSV.
2. Trên Streamlit Cloud, chọn repository, branch và `Main file path = app.py`.
3. `requirements.txt` được cài tự động khi deploy; kiểm tra log nếu môi trường khác Python 3.12/3.13.

### Quy trình xem kết quả

- Mục **RCM**: phần mềm chạy 9 mô hình trên dữ liệu đo RCM (không phải nhãn tuổi thọ).
- Mục **Thiết kế ngược RCM**: nhập ngưỡng RCM, xuất danh sách ứng viên và kết quả đo nếu có.
- Mục **Tuổi thọ AI** > **Dự báo sống sót & bài toán ngược**: tải CSV đã thu thập có cấu trúc đúng. Phần mềm sẽ tách theo `limit_state` và `exposure_protocol`, đánh giá số mẫu/sự kiện/nhóm nguồn, rồi chạy survival. Nếu đủ nguồn, kiểm định holdout vượt các ngưỡng tối thiểu thì mới cho dự báo và bài toán ngược tuổi thọ.
- Mục **Nguồn thí nghiệm quốc tế**: xem 12 tài liệu, DOI và 19 bản ghi gia tốc; 19 bản ghi không được tự coi là 19 tuổi thọ hiện trường.

## Survival ML (3 mô hình có xử lý mẫu kiểm duyệt)

- **Cox Proportional Hazards (Ridge)**: tối ưu hàm likelihood bán phần với kiểm duyệt phải, ước lượng nguy cơ nền Breslow.
- **Random Forest Survival**: mô hình rừng ngẫu nhiên dự báo hazard theo khoảng thời gian (pooled discrete-time logistic hazard); mẫu kiểm duyệt giữa kỳ không được gán nhãn âm cho kỳ chưa quan sát đủ.
- **Gradient Boosting Survival**: gradient boosting phân loại nguy cơ theo thời kỳ, xử lý kiểm duyệt như trên.
- **Tự động lựa chọn**: GroupKFold trên train, đánh giá Harrell C-index và IPCW Brier. Giữ riêng một nhóm nguồn làm holdout bằng GroupShuffleSplit. Tuyệt đối không dùng kết quả holdout để chọn mô hình.
- **Kiểm tra đầu ra**: ngưỡng số mẫu/sự kiện/nguồn; holdout tối thiểu C-index ≥0.55, ≥20 cặp có thể so sánh, IPCW Brier thấp hơn Kaplan-Meier baseline. Đây là **quy tắc sàng lọc nội bộ**, không phải tiêu chuẩn chứng nhận kỹ thuật, không chứng minh khả năng dự báo tuổi thọ công trình.
- **Không ngoại suy**: chỉ dự báo trong khoảng từng biến đầu vào và thời gian thực nghiệm đã quan sát ở train, môi trường đã gặp. Điều kiện này vẫn *không* đảm bảo mọi điểm nội suy đa chiều là đáng tin.

### CSV chuẩn để huấn luyện

Tải file mẫu **chỉ có tiêu đề** trong phần mềm hoặc xem `CloriLife_v10_MAU_DU_LIEU_SURVIVAL.csv`.

| Cột | Bắt buộc | Giải thích |
|---|---|---|
| `fly_ash_pct` | Có | Tro bay theo % chất kết dính |
| `water_binder_ratio` | Có | N/CKD |
| `cover_mm` | Có | Lớp bảo vệ bê tông mm |
| `exposure_class` | Có | Nhóm môi trường trong quy trình khảo sát |
| `observed_duration_years` | Có | Thời gian đến sự kiện **hoặc** thời gian theo dõi cuối khi chưa xảy ra sự kiện |
| `event_observed` | Có | `1`: đã xảy ra, `0`: chưa xảy ra khi kết thúc theo dõi (censored) |
| `source_group` | Có | Mã một nghiên cứu/nhóm nguồn độc lập (không được tách một nghiên cứu thành nhiều nguồn giả) |
| `limit_state` | Có | Tên một sự kiện chuẩn, ví dụ `corrosion_initiation` |
| `exposure_protocol` | Có | `natural_marine` hoặc `accelerated_*`, không trộn khi học |
| `source_reference` | Có | URL/DOI/tài liệu gốc của từng kết quả |
| `d_rcm_28_m2_s` | Tùy chọn | Chỉ điền nếu có RCM đo thực nghiệm ghép **đúng mẫu**. Nếu thiếu ở bất cứ dòng nào, toàn mô hình không dùng RCM làm biến. |

**Điểm quan trọng về thời gian:** Nếu dữ liệu là gia tốc, có thể đổi giờ/ngày sang đơn vị năm để dùng chung đơn vị *thời gian tính toán*, nhưng kết quả vẫn là **thời gian thí nghiệm gia tốc**, không được nhân một hệ số tự đặt để công bố tuổi thọ hiện trường. Dữ liệu quan sát ăn mòn chỉ tại mốc kiểm tra tiếp theo (interval censored) chưa được xử lý bởi các mô hình v1.0; không điền sai thành ngày ăn mòn chính xác.

### Số liệu thực nghiệm đã được đưa vào gói

- `real_rcm_flyash_215.csv` — dữ liệu RCM, không có tuổi thọ theo năm.
- `research_sources/RCM_8_TABLES_1268_RESEARCH.csv` — dữ liệu RCM của 8 bảng đã lập nguồn từ v0.9 (không phải bản ghi tuổi thọ).
- `research_sources/RCM_FLYASH_SLAG_557_RESEARCH.csv` — 557 bản ghi tro bay + xỉ, là tập con/nhóm RCM không dùng thay tuổi thọ.
- `corrosion_initiation_19_groups.csv` — 19 ghi nhận ăn mòn gia tốc từ ba nghiên cứu, có thời gian gốc theo ngày/giờ, xuất xứ và chú thích. Một số là số liệu trung bình từng cấp phối, không phải 19 mẫu thử độc lập cùng quy trình.
- `research_sources/ACCELERATED_19_AUDIT_ONLY.csv` — dạng quy đổi đơn vị để **kiểm tra cấu trúc**, không đủ để mô hình hiện trường học; không có RCM đo ghép.
- `research_sources/VERIFIED_STUDIES_2026.csv` — phân loại 12 nguồn, chỉ rõ loại kết quả và mức độ đủ điều kiện để huấn luyện tuổi thọ.
- `research_sources_14.csv` — danh sách nghiên cứu RCM và tuổi thọ từ bản v0.9.

Tài liệu có thời gian phơi nhiễm 20 hoặc 36 năm nhưng chỉ đo **nồng độ clorua** không được gán bằng 20/36 năm tuổi thọ. Nghiên cứu Florida (1995–2016) báo cáo một số thanh thép bắt đầu ăn mòn sau 15,6–17 năm, nhưng phải trích đúng dữ liệu từng mẫu, mốc giám sát và thành phần cấp phối mới có thể tạo bản ghi huấn luyện. Danh mục DOI có trong CSV kèm theo.

## Kiểm thử & tính năng chưa được xác nhận

```bash
python -m pip install pytest
python -m pytest -q tests
```

Các test dữ liệu tự sinh nằm **chỉ trong `tests/`** để kiểm tra mã, không phải thí nghiệm khoa học và không xuất hiện trong tệp đầu vào sản xuất. Có các test cho kiểm duyệt phải, ngoại suy, chia nguồn, mô hình theo thời gian, các phép đo C-index/Brier, bài toán ngược RCM và tuổi thọ.

**Giới hạn kỹ thuật:** Chưa có dữ liệu thời gian bắt đầu ăn mòn ở hiện trường có đủ dữ liệu cùng mẫu và nhóm nguồn độc lập để xác nhận R²/C-index thực tế của mô hình tuổi thọ. Chưa đánh giá lan truyền sai số khi dùng RCM dự báo bằng AI làm đầu vào mô hình tuổi thọ. Chưa có mô hình interval-censoring và cạnh tranh trạng thái giới hạn; không có bảo đảm 50/75/100 năm đối với công trình thực. Không dùng kết quả làm cơ sở lựa chọn chiều dày bảo vệ, chấp thuận vật liệu hoặc quyết định an toàn kết cấu.

## Nguồn công bố tiêu biểu

- Dữ liệu RCM 215 mẫu: https://doi.org/10.5061/dryad.66t1g1k6j
- 2016 Mỹ, một số mẫu ăn mòn sau 15,6–17 năm: https://doi.org/10.14359/51689184
- 2014 Thụy Điển, kiểm tra sau 13 và 20 năm: https://research.chalmers.se/en/publication/211415
- 2026 Pháp, nồng độ clorua tới 36 năm (không phải nhãn ăn mòn): https://doi.org/10.57745/CMJUI3
- 2026 Zenodo, ngưỡng clorua/pH/điện thế (không phải thời gian): https://doi.org/10.5281/zenodo.21987599

**Bản v0.9 gốc**: `README_v0_9_LUU_TRU.md`; thuật toán và giao diện tương thích đã được giữ trong `lifespan_ui_legacy.py`.
