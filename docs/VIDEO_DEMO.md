# Ứng dụng phân tích video trên máy cá nhân

Ứng dụng nhận video MP4, MOV, AVI hoặc MKV, chạy model trên CPU, hiển thị tiến độ và tạo video MP4 với bounding box, nhãn và điểm tin cậy. Video kết quả không có âm thanh. Đây là bản demo hỗ trợ con người xem xét, không phải hệ thống ra quyết định an ninh. Dự đoán có thể sai hoặc bỏ sót; lớp `potentially_dangerous_person` mô tả ngữ cảnh hình ảnh, không xác định ý định hay mức nguy hiểm thực tế.

## Chạy trên Windows

Mở PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\python.exe -m pip install -r video_app/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn video_app.app:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Nếu chưa có `.venv`, chạy `python -m venv .venv` trước. Truy cập `http://127.0.0.1:8000`, chọn video rồi nhấn **Bắt đầu phân tích**. Chờ xử lý xong để xem hoặc tải kết quả. Nhấn `Ctrl+C` để dừng ứng dụng. Không cần Azure, Docker hoặc cài FFmpeg riêng: gói `imageio-ffmpeg` cung cấp FFmpeg.

Ứng dụng dành cho một người dùng trên máy cá nhân và không có cơ chế đăng nhập. Chỉ chạy trên `127.0.0.1`; không mở ra mạng hoặc Internet. Các video đang xử lý và kết quả được lưu trong thư mục tạm của hệ điều hành; khi dừng ứng dụng bình thường, thư mục này được xóa. Nếu tiến trình bị tắt đột ngột, tệp tạm có thể còn lại trong thư mục có tiền tố `video-review-`. Tải kết quả về trước khi dừng ứng dụng. Chỉ một video được xử lý tại một thời điểm; các video khác chờ. Thời gian xử lý phụ thuộc độ dài video và CPU.

Mặc định không giới hạn kích thước hay thời lượng video; giải mã hỗ trợ tối đa 3840×2160 và 120 FPS. Có thể đặt `MAX_VIDEO_MB` hoặc `MAX_VIDEO_SECONDS` thành số nguyên dương trước khi chạy để giới hạn. Tệp lớn cần đủ dung lượng trống trên ổ tạm. Nội dung có phần mở rộng hợp lệ nhưng bị hỏng hoặc sử dụng codec không được hỗ trợ sẽ báo lỗi.

## Model và kiểm thử

Model mặc định là `kaggle-output/model-comparison/yolo11s-fit-within-640/weights/best.pt`, SHA-256 `be638b965b2978e8abcd5ef506e63ba19743fd88a0582d2a96ffdc390a9ba3f5`. Checkpoint có lớp 0 `normal_person` và lớp 1 `potentially_dangerous_person`. Có thể đặt `MODEL_PATH` để dùng checkpoint khác có cùng ánh xạ lớp. Nguồn gốc, quyền sử dụng dữ liệu và giấy phép checkpoint chưa được xác minh; cần kiểm tra trước khi phân phối. Chỉ tải checkpoint đáng tin cậy.

Chạy kiểm thử ứng dụng bằng `python -m unittest discover -s video_app/tests -v` sau khi cài `httpx`.
