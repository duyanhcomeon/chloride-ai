# CloriLife-AI v1.0 — Thay đổi so với v0.9

- Thêm mô-đun `survival_ml.py`: Cox PH có regularization; Random Forest và Gradient Boosting học hazard theo kỳ, xử lý mẫu chưa xảy ra sự kiện (right-censored).
- Bổ sung C-index Harrell, IPCW Brier và đường chuẩn Kaplan-Meier; tách nguồn nghiên cứu giữa train/holdout; CV theo nguồn.
- Bổ sung xác suất chưa ăn mòn ở mốc thời gian, đường cong survival, ước lượng thời gian trung vị nếu đủ dữ liệu, CSV xuất kết quả.
- Thêm bài toán ngược theo thời gian đến sự kiện và xác suất đạt mục tiêu, tìm kiếm bên trong **bao lồi** cấp phối cùng môi trường từ train.
- Tách điều kiện phơi nhiễm tự nhiên khỏi gia tốc, không nội suy trực tiếp thí nghiệm điện trường/nồng độ muối cao thành năm ngoài hiện trường.
- Giữ các mô hình và bài toán ngược RCM đã có của v0.9, đồng thời giữ tab hồi quy thời gian đủ nhãn cho người dùng cũ.
- Tích hợp 12 nguồn thí nghiệm có DOI/URL, 19 điều kiện thí nghiệm ăn mòn gia tốc với bảng chuẩn hóa phục vụ kiểm tra, 1.268 dòng RCM từ 8 bảng và 557 mẫu FA+SG. Không tự tạo dữ liệu thời gian ăn mòn.
- Thêm CSV mẫu trống, hướng dẫn chạy trên Windows, README mới và kiểm thử hồi quy + survival.

**Chưa thể hoàn tất bước xác nhận khoa học dự báo tuổi thọ hiện trường:** cần tập dữ liệu có thời gian bắt đầu ăn mòn từng mẫu, lớp bảo vệ, cấp phối, điều kiện môi trường, thời gian theo dõi và nguồn độc lập; hiện không thể báo cáo một số năm tuổi thọ có độ chính xác được chứng minh.
