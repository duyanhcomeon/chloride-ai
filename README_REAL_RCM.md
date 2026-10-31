# Nguồn dữ liệu RCM thực nghiệm và nguyên tắc sử dụng — CloriLife AI v0.7.1

**Dữ liệu:** Yang, Zhu và Wei; Excel thí nghiệm RCM được công bố tại https://doi.org/10.5061/dryad.66t1g1k6j.

- 215 bản ghi bê tông tro bay từ Table 5 trong tệp Excel do người dùng cung cấp, có `source_reference`, `source_group`, `source_excel_row` để truy xuất.
- Biến đầu vào đã đo/ghi nhận: tỷ lệ tro bay, N/CKD, loại xi măng, cấp tro bay.
- Biến mục tiêu: hệ số di chuyển ion clorua bằng phương pháp RCM (m²/s), **không** phải D_app và **không** phải tuổi thọ.
- Thiếu dữ liệu ở một số bản ghi cấp tro bay được mã hóa tạm là `khong_ghi` trong mô hình, không được suy diễn cấp vật liệu.
- Chia dữ liệu theo nguồn: 20% nhóm nguồn làm holdout; chọn trong 9 thuật toán bằng 5-fold GroupKFold trên phần train.
- Module `inverse_design.py` chỉ dùng `training_rows`, không dùng RCM đo của tập holdout để gợi ý thiết kế; có thể dùng MAE CV train làm biên sàng lọc.
- Bài toán ngược không có nghiệm duy nhất. Mọi gợi ý từ mô hình cần thí nghiệm RCM xác nhận. Không được quảng cáo đây là cấp phối bê tông hoàn chỉnh khi dữ liệu thiếu khối lượng xi măng, cốt liệu, phụ gia, cường độ, độ sụt.

Để xem hướng dẫn triển khai, mô hình, ba chế độ tìm nghiệm và giới hạn khoa học, xem `README.md`.
