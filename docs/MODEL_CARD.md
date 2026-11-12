# Thông tin mô hình

## Mục tiêu

Phát hiện và phân loại bảy nhóm vật thể điện tử: pin, chuột, bàn phím, laptop, điện thoại, bảng mạch và bộ sạc. Đây là bản phục vụ nghiên cứu và trình diễn nhận diện hình ảnh.

## Trọng số

| Thành phần | File | Kiến trúc | Thông tin xác nhận |
|---|---|---|---|
| Detector | `train_yolo/best6.pt` | YOLO26s, 7 lớp | Metadata checkpoint: Ultralytics 8.4.37 |
| Classifier | `train_cnn/best_convnext_research5.pth` | ConvNeXt Tiny, đầu ra 7 lớp | Nạp được bằng kiến trúc torchvision, state dict khớp đầy đủ |

SHA-256 và dung lượng chính xác nằm trong [`models_manifest.json`](../models_manifest.json). Trọng số được giữ nguyên nội dung từ dự án gốc. Chỉ nạp checkpoint từ nguồn tin cậy; định dạng checkpoint YOLO sử dụng cơ chế tuần tự hóa của PyTorch.

## Thứ tự nhãn

```text
0 battery
1 computer mouse
2 keyboard
3 laptop
4 mobile
5 pcb
6 phone charger
```

Nhãn YOLO được đọc trực tiếp từ checkpoint. Thứ tự CNN được khôi phục từ thứ tự `ImageFolder` và danh sách lớp trong notebook; checkpoint CNN không chứa bản ánh xạ nhãn riêng. Nếu huấn luyện lại, cần lưu kèm thứ tự lớp và cập nhật cấu hình.

## Pipeline suy luận

1. YOLO nhận frame BGR từ OpenCV, ngưỡng phát hiện mặc định 0,25.
2. Các hộp được giới hạn trong kích thước frame; hộp rỗng bị loại bỏ.
3. Giữ nhãn YOLO khi điểm đạt ngưỡng chuyển mặc định 0,6.
4. Với phát hiện dưới ngưỡng chuyển, crop được đổi BGR sang RGB, resize 224 × 224 và chuẩn hóa theo mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
5. ConvNeXt trả nhãn có softmax lớn nhất. Hộp vẫn là vị trí do YOLO tạo.

`confidence` là điểm của thành phần quyết định nhãn. `detection_confidence` luôn là điểm YOLO; `classification_confidence` chỉ có khi CNN được gọi. Hai điểm không được hiệu chỉnh chung và không nên diễn giải như xác suất đúng đã được kiểm chứng.

## Dữ liệu và đánh giá

Notebook liên kết dự án Roboflow `hoainam-pusiu/nhan-dien-rac-dien-tu-ehsyl`. Dataset ảnh không được phân phối trong kho. Notebook dùng nhiều phiên bản dataset và nhiều checkpoint cũ; chưa có manifest hoàn chỉnh nối từng checkpoint với một split dữ liệu cụ thể.

`train_yolo/results.csv` chứa nhật ký validation 83 epoch. Chưa đánh giá lại checkpoint trên một tập test cố định trong bước đóng gói. Không công bố con số accuracy/mAP cho bản hybrid hiện tại khi chưa có đánh giá tương ứng.

## Giới hạn

- Bảy lớp cố định; mô hình có thể gán một vật thể ngoài danh sách vào lớp đã biết.
- CNN chỉ xử lý vật thể YOLO đã tìm được. Không có hộp YOLO thì không có kết quả nhận diện.
- Ánh sáng, góc chụp, che khuất, vật thể nhỏ và nền khác dữ liệu huấn luyện có thể làm giảm chất lượng.
- Tốc độ webcam phụ thuộc phần cứng; chưa có benchmark FPS được đo cho bản xuất bản này.
- Không xác nhận vật thể còn hoạt động hay đã trở thành rác thải, mức độ nguy hại hoặc cách xử lý chất thải.
- Notebook lịch sử có biến thể phân loại toàn ảnh khi YOLO bỏ sót; CLI hiện tại không dùng biến thể đó.
