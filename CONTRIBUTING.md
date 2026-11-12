# Đóng góp

Tạo môi trường Python, cài `requirements.txt`, rồi chạy:

```bash
python -m unittest discover -s tests -v
python scripts/check_repository.py
```

Khi thay đổi mô hình, kiểm tra thêm bằng ảnh có nhãn và ghi rõ dataset version, split, phần cứng, tham số, thứ tự nhãn và checkpoint. Khi cập nhật trọng số, dùng Git LFS và cập nhật SHA-256 trong `models_manifest.json`.

Không commit khóa API, `.env`, dataset, môi trường Python hoặc output chạy thử. Trước khi commit notebook, xóa output và execution count. Mô tả thay đổi cần nêu vấn đề được giải quyết và cách kiểm tra thực tế.
