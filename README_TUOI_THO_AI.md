# Dự báo thời gian bắt đầu ăn mòn bằng AI (Survival v1.0)

Đây là mô-đun thời gian đến sự kiện, khác với mô-đun RCM. Vui lòng đọc **README.md** để có hướng dẫn đầy đủ, chú thích nguồn và giới hạn khoa học.

1. Có 3 thuật toán: Cox PH Ridge; pooled discrete-time Random Forest hazard; pooled discrete-time Gradient Boosting hazard. Cả 3 xử lý kiểm duyệt phải khi thử nghiệm kết thúc nhưng chưa xảy ra ăn mòn.
2. Không dùng Fick. Không coi 215 dòng RCM là 215 tuổi thọ. Không trộn `limit_state` hoặc `exposure_protocol`.
3. Phải có CSV theo mẫu `CloriLife_v10_MAU_DU_LIEU_SURVIVAL.csv`; tối thiểu 40 dòng, 8 nhóm nghiên cứu, 12 sự kiện cho mỗi tổ hợp trạng thái giới hạn / quy trình phơi nhiễm.
4. Chia train/test theo nguồn nghiên cứu; dùng GroupKFold chọn mô hình; giữ holdout chỉ để đánh giá cuối. C-index và IPCW Brier được tính với kiểm duyệt phải; IPCW cần giả định phù hợp về kiểm duyệt.
5. Khi holdout không đạt sàng lọc nội bộ, ứng dụng **khóa dự báo và bài toán ngược tuổi thọ**. Khi đạt, chỉ dự báo trong miền điều kiện và thời gian đã có thực nghiệm.
6. Không dùng thời gian gia tốc như tuổi thọ hiện trường. Dữ liệu chỉ có trạng thái ăn mòn ở hai kỳ kiểm tra là interval censored; mô-đun hiện chưa hỗ trợ interval censoring.
7. Bài toán ngược tìm các phương án trong bao lồi 2D (tro bay/N-CKD) của mẫu train cùng môi trường và lân cận chuẩn hóa. Nếu mô hình survival có dùng RCM, cần mô hình RCM hợp lệ để dự báo lại hệ số RCM cho **từng cấp phối**.

**Hiện trạng v1.0:** Đã hoàn thành phần mềm và kiểm thử chức năng. Chưa có bộ dữ liệu tuổi thọ tự nhiên đủ nhãn đã trích xuất và xác minh; vì vậy không có kết quả tuổi thọ 50/75/100 năm được chứng minh. Bộ danh mục 12 nghiên cứu và 19 bản ghi thí nghiệm gia tốc ở thư mục `research_sources/` phục vụ tiếp tục thu thập và kiểm tra dữ liệu, không phải tệp huấn luyện tuổi thọ hiện trường.
