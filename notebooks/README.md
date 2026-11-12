# Notebook nghiên cứu

| Notebook | Nội dung |
|---|---|
| [yolo26s_dataset_v7.ipynb](yolo26s_dataset_v7.ipynb) | YOLO26s, Roboflow v7, huấn luyện/đánh giá, crop dữ liệu và thử CNN |
| [cnn_comparison_and_fusion.ipynb](cnn_comparison_and_fusion.ipynb) | So sánh ConvNeXt, ResNet, EfficientNet và nhiều cấu hình YOLO + CNN |

## Sử dụng

1. Cài `pip install -r requirements-research.txt` từ thư mục gốc.
2. Chạy `jupyter lab` hoặc mở notebook trong Colab/Kaggle.
3. Đặt `ROBOFLOW_API_KEY`; nếu chưa đặt, cell tải dữ liệu sẽ hỏi khóa bằng `getpass`.
4. Kiểm tra phiên bản dataset, đường dẫn, checkpoint và GPU trước khi chạy từng phần.

Notebook được giữ như hồ sơ nghiên cứu, không phải pipeline có thể “Run All” trên mọi máy. Hai cell train YOLO liên tiếp là bản lặp từ thí nghiệm; chọn cell cần chạy. Những đường dẫn Google Drive và checkpoint phụ trong notebook cần được cung cấp riêng.

Các output cũ đã được xóa để tránh công khai thông tin xác thực và để Git diff dễ xem. Kết quả nhận diện hiện tại chạy qua `python main.py --source ...` ở thư mục gốc, không phụ thuộc notebook.
