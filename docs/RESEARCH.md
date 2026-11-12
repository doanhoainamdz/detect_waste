# Nguồn nghiên cứu và khả năng tái lập

## Đề tài

Ứng dụng học sâu trong nhận dạng và phân loại rác thải điện tử từ hình ảnh giám sát, sử dụng YOLO kết hợp mạng nơ-ron tích chập.

Kho này đóng gói phần mã và trọng số để chạy nhận diện, đồng thời giữ các notebook thí nghiệm hiện có. Các bản nháp Word, PDF báo cáo, phiếu đăng ký, môi trường Python và cache từ thư mục gốc không được đưa vào bản xuất bản.

## Thành phần được giữ

| Nguồn | Vai trò |
|---|---|
| `train_yolo/best6.pt` | Detector YOLO26s đã huấn luyện |
| `train_cnn/best_convnext_research5.pth` | Bộ phân loại ConvNeXt Tiny |
| `train_yolo/results.csv` | Nhật ký huấn luyện YOLO 83 epoch |
| `notebooks/yolo26s_dataset_v7.ipynb` | Thí nghiệm YOLO26s, dataset Roboflow v7; có cả phần thử CNN |
| `notebooks/cnn_comparison_and_fusion.ipynb` | ConvNeXt, EfficientNetV2, ResNet và nhiều biến thể kết hợp |

Notebook được đổi tên cho dễ đọc, bỏ output và execution count, xóa metadata phụ, chuyển khóa Roboflow sang biến môi trường hoặc nhập ẩn. Các cell thí nghiệm và tham số lịch sử được giữ; không chuẩn hóa kết quả của nhiều lần chạy thành một thí nghiệm duy nhất.

## Nhật ký huấn luyện

Biểu đồ trong README đọc trực tiếp bốn cột validation precision, recall, mAP50 và mAP50-95 trong CSV. CSV có 83 dòng epoch. Các chỉ số này là lịch sử có sẵn, chưa được tái lập trong bước xuất bản và chưa xác minh quan hệ với `best6.pt`.

Không suy ra accuracy phân loại CNN từ mAP detector. Các số liệu điền tay trong cell so sánh notebook là ghi chú thí nghiệm, không được sử dụng làm kết quả xác nhận của kho.

## Những điểm cần cấu hình khi nghiên cứu tiếp

- Dataset không nằm trong kho; cần quyền truy cập Roboflow và chọn đúng phiên bản.
- Notebook CNN có các đường dẫn dataset v7, v9, v16 và v21, cũng như nhiều checkpoint trên Google Drive. Một số checkpoint đó không có trong bản hiện tại.
- Notebook YOLO chứa hai cell `model.train(...)` giống nhau. Chọn một cell nếu chỉ muốn huấn luyện một lượt.
- Một số cell yêu cầu GPU hoặc hai GPU (`device=[0,1]`). Cần điều chỉnh theo phần cứng.
- Một số cell đánh giá phụ thuộc biến từ cell trước và không thích hợp để chạy tuần tự bằng “Run All”.
- Cần lưu `class_to_idx`, dataset version, seed, split và cấu hình phần cứng cùng checkpoint khi huấn luyện lại.

## Kiểm tra chất lượng mã

Unit test kiểm tra chuyển nhánh YOLO/CNN, crop ngoài biên, ảnh lỗi, chuẩn hóa và chuyển BGR sang RGB, truyền thiết bị/ngưỡng vào YOLO, và nhận biết Git LFS pointer. Kiểm tra kho xác nhận notebook không chứa output hay khóa API viết cứng và xác minh checksum trọng số khi được yêu cầu.

Những kiểm tra này xác nhận luồng chương trình và khả năng nạp mô hình, không thay thế đánh giá chất lượng nhận diện trên tập test có nhãn.
