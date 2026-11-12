# Trọng số CNN

`best_convnext_research5.pth` là state dict ConvNeXt Tiny với đầu ra 7 lớp, quản lý bằng Git LFS. File thật có dung lượng 111.371.607 byte, vượt giới hạn file Git thông thường của GitHub.

Chạy `git lfs pull` để tải. Kiến trúc được dựng trong `module/cnn_classifier.py`; trọng số tiền huấn luyện ImageNet không cần tải thêm khi nạp checkpoint này. Thứ tự lớp nằm trong `module/config.py` và `models_manifest.json`.
