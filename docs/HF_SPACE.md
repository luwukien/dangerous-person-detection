# Hugging Face ZeroGPU demo

Space trong `hf_space/` là một demo Gradio độc lập, không thay thế ứng dụng FastAPI chạy cục bộ. Chỉ tải video MP4, MOV, AVI hoặc MKV tối đa 20 MB và 15 giây. Kết quả MP4 không có âm thanh. Đây là hỗ trợ xem xét, không phải hệ thống ra quyết định an ninh; dự đoán có thể sai hoặc bỏ sót.

## Chuẩn bị trước khi công khai

- Kiểm tra nguồn gốc và quyền phân phối dữ liệu huấn luyện, checkpoint, các thư viện và nội dung video mẫu. Các quyền này chưa được xác minh trong dự án. Không công khai checkpoint hoặc video khi chưa có quyền.
- Checkpoint hiện có tại `model-comparison/yolo11s-fit-within-640/weights/best.pt`, SHA-256 `be638b965b2978e8abcd5ef506e63ba19743fd88a0582d2a96ffdc390a9ba3f5`. Chỉ sử dụng checkpoint đáng tin cậy, có các lớp 0 `normal_person`, 1 `potentially_dangerous_person`.
- Cần tài khoản Hugging Face đủ điều kiện tạo Space ZeroGPU. Điều kiện, hàng đợi và quota có thể thay đổi; xem [tài liệu ZeroGPU](https://huggingface.co/docs/hub/spaces-zerogpu).

## Tạo Space

1. Tạo Space mới tại Hugging Face, chọn **Gradio SDK** và phần cứng **ZeroGPU** (có thể chọn trong Settings nếu chưa thấy khi tạo). Chọn private trong lúc chuẩn bị nếu tài khoản cho phép.
2. Upload nội dung của `hf_space/` lên **thư mục gốc của Space**: `README.md`, `app.py`, `video_processing.py`, `requirements.txt`. Không cần upload thư mục `tests`.
3. Sau khi xác minh quyền phân phối, upload checkpoint nêu trên thành `best.pt` ở thư mục gốc của Space. Không upload dataset, token hoặc thông tin cá nhân. Space sẽ tự build; kiểm tra logs nếu có lỗi. Đặt `MODEL_PATH` trong Space nếu chọn một đường dẫn checkpoint khác.
4. Thử một video ngắn được phép sử dụng. Xác nhận video đầu ra phát và tải được, đo thời gian từ lúc gửi tới lúc xong, và kiểm tra trong logs rằng GPU hoạt động. Nếu quá thời gian GPU hoặc quota, giảm thời lượng video và thử lại. ZeroGPU có thể xếp hàng hoặc từ chối khi hết quota; không bảo đảm luôn có GPU.
5. Chỉ chuyển sang public sau khi xác minh quyền, nội dung và luồng hoạt động. Thêm URL Space và video quay màn hình dự phòng vào README/CV. Chưa có link công khai trong repository này.

Video tải lên và kết quả được lưu tạm trên máy chủ. Cache Gradio được cấu hình xóa theo chu kỳ; kết quả cũ hơn một giờ được dọn khi có yêu cầu tiếp theo, và dữ liệu tạm có thể còn lại khi tiến trình dừng bất thường. Không dùng video nhạy cảm. Không có cam kết lưu trữ lâu dài.

## Kiểm thử cục bộ

Cài dependencies trong `hf_space/requirements.txt`, cùng phiên bản Gradio trong cấu hình Space, rồi chạy `python hf_space/app.py`. Ngoài Space, demo dùng CPU; checkpoint mặc định là `hf_space/best.pt` hoặc đường dẫn `MODEL_PATH`. Chạy `python -m unittest discover -s hf_space/tests -v` để kiểm tra xử lý video. GPU ZeroGPU và hàng đợi chỉ xác minh được trên Space.
