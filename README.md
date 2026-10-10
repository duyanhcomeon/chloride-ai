# Chloride-AI MVP v0.1 — BTCT sử dụng tro bay

Ứng dụng chạy tại máy cục bộ, dự báo **thời gian đạt ngưỡng clorua tại cốt thép**, KHÔNG dự báo toàn bộ tuổi thọ kết cấu.

## Cài đặt (Windows / macOS / Linux)
```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Ứng dụng thường mở tại http://localhost:8501.

## Kiến trúc
```
CSV số liệu thí nghiệm → ml.py (RandomForest, log10 D28) ┐
Nhập D28 đã đo ──────────────────────────────────────────┤
Thông số kết cấu & điều kiện Cl⁻ ────────────────────────┤
                                                       physics.py (Fick + ∫Ddt)
                                                               ↓
                                                     app.py (Streamlit + CSV)
```

## Cách thử
1. Khởi chạy ứng dụng. Chọn `Nhập hệ số đã đo` và nhập D28 có nguồn đo hoặc giá trị minh họa.
2. Nhập lớp bảo vệ, nồng độ bề mặt, nồng độ ban đầu, ngưỡng, m và thời gian phân tích.
3. Xem đồ thị và tải CSV.
4. Muốn thử AI, sử dụng `data_demo_SYNTHETIC.csv` đi kèm gói mã nguồn. **Đây là dữ liệu mô phỏng, không phải thí nghiệm**, không được dùng đánh giá khoa học hoặc công trình.
5. Chạy kiểm thử: `python -m pytest -q`.

## Dữ liệu huấn luyện thực
CSV cần đủ `fly_ash_pct,water_binder_ratio,curing_days,compressive_strength_mpa,temperature_c,d28_m2_s`.
D phải là hệ số khuếch tán biểu kiến ở 28 ngày (m²/s), được đo/chuẩn hóa bằng quy trình tương thích. Trường `curing_days` là thông tin bảo dưỡng/tuổi mẫu liên quan nhưng **không thể thay thế hồ sơ thí nghiệm**. Cần bổ sung phân nhóm theo cấp phối, phòng thí nghiệm, nguồn tro bay, phương pháp thử, môi trường khi nâng cấp.

## Giới hạn & cần nâng cấp
- Mặc định m=0.35 và tất cả thông số màn hình là **giả định minh họa**, không được coi là các hệ số chuẩn.
- Mô hình vận chuyển một chiều lý tưởng, nồng độ bề mặt không đổi, không nứt, bỏ qua tương tác ẩm–nhiệt, chu kỳ ướt–khô, ràng buộc Cl⁻ và ảnh hưởng CO₂.
- Random train/test split mang tính thử nghiệm phần mềm; nghiên cứu cần grouped/nested validation, định lượng bất định và dữ liệu thực địa.
- Chưa dự báo giai đoạn phát triển ăn mòn và tổn thất khả năng chịu lực.
- Không sử dụng để nghiệm thu, thiết kế, đánh giá an toàn công trình khi chưa hiệu chuẩn và kiểm chứng độc lập.

## Tài liệu khoa học
- ASTM C1556-25 (kiểm tra hệ số khuếch tán biểu kiến, phạm vi áp dụng và lưu ý tuổi mẫu): https://store.astm.org/standards/c1556
- Time dependent models of apparent diffusion coefficient and surface chloride for chloride transport in fly ash concrete (2013): https://www.sciencedirect.com/science/article/pii/S0950061812006381
