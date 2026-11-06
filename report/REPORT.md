# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab; `rank`=1 ; `score`=0.510915; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh**  trong ảnh? Vì Không có bbox , nên không chỉ ra vật nào ở vị trí nào. Và ảnh có nhiều đối tượng
3. class list **đi kèm taxonomy/checkpoint**,File yolo11n-cls.pt được huấn luyện sẵn trên bộ dữ liệu có danh sách lớp cố định (ImageNet-1K, 1000 lớp)
4. Vì ghi đủ ba thứ giúp kết quả rõ ràng, kiểm tra lại được và không bịnhầm giữa các taxonomy `class_id` : để đối chiếu chính xác và xử lý bằng máy,`class_name` giúp người đọc hiểu nhanh,`taxonomy_name` cho biết class_id thuộc danh sách nào
5. Guideline phải nêu quy tắc chọn đúng một lớp, ví dụ như : chủ thể chính | Cách xử lý khi nhiều vật có tầm quan trọng ngang nhau | có lớp nào phù hợp trong taxonomy thì làm gì
6. Vì `score` không phải ground truth vì:
- Ground truth là nhãn do **người** gán theo guideline; `score` chỉ là mức "tin" của model vào dự đoán của chính nó, không chứng minh đúng hay sai.
- Model có thể **tự tin nhưng sai**: ảnh `traffic` có score `cab` = 0.51, nhưng nội dung chính là nhiều xe buýt, nên `cab` chưa chắc đúng.
- Nếu lấy score làm ground truth thì thành **vòng lặp tự xác nhận** (dùng câu trả lời của model để chấm model) và đưa lỗi của model vào dữ liệu huấn luyện.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= bowl; `score`=0.719063 ; `bbox_xyxy`=[32.65, 342.12, 100.16, 384.93] ; `bbox_width`=67.51 ; `bbox_height`=42.81
2. Lớp `bowl`, score 0.72; box từ góc trên-trái (32.65, 342.12) đến góc dưới-phải (100.16, 384.93); rộng khoảng 67.5 px, cao khoảng 42.8 px.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). 
- Ngưỡng 0.20: 17 prediction (person, bowl, oven, cup, spoon, potted plant, dining table, bottle...).
- Ngưỡng 0.60: 6 prediction (person ×2, bowl ×2, oven ×2).
Hạ ngưỡng từ 0.60 xuống 0.20 thì số prediction tăng từ 6 lên 17: thêm nhiều box điểm thấp (spoon, potted plant, dining table, bottle...) mà ngưỡng cao đã loại.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn)thì phải giữ thêm nhiều box điểm thấp, trong đó có nhiều box sai hoặc lệch (false positive). Reviewer phải kiểm tra từng box: xóa box thừa, sửa nhãn sai, chỉnh box cho sát, nên tốn công hơn. Đổi lại, ít bỏ sót vật hơn.
5. Viết **một quy tắc box** : Mỗi vật thuộc taxonomy được vẽ một box chữ nhật sát quanh phần nhìn thấy của vật đó, kể cả khi bị che một phần; không vẽ box cho vật quá nhỏ hoặc quá mờ để nhận ra, và không vẽ hai box cho cùng một vật.
6. Trên ảnh `kitchen`: bàn tay/cánh tay ở mép trái bị khung ảnh cắt, chỉ còn một phần nhỏ (box person điểm thấp).Tôi sẽ escalate cho Lab Coach: guideline chưa nói rõ chỉ thấy một phần cơ thể thì có gán "person" không, nên tự đoán sẽ khiến mỗi annotator làm một kiểu. Khi có câu trả lời, mới viết thành quy tắc chung.
## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-011 ; `class_name`= spoon ; `score`= 0.359147 ; số điểm polygon=`polygon_point_count`=37 ; 3–5 điểm đầu của `polygon_xy`=[155.0, 170.0], [155.0, 177.0], [156.0, 178.0], [156.0, 179.0], [161.0, 184.0]
2. So với box, polygon cho thêm thông tin về đường biên thật của vật (12 điểm, đáy bo tròn), nên biết được hình dạng và diện tích thực, loại bỏ phần nền thừa.
3. `instance_id`  dùng để phân biệt từng đối tượng riêng trong cùng một ảnh, mỗi vật có một mask riêng. Nó không phải class ID: class_id (45) là mã của lớp "bowl", mọi cái bát đều chung mã này. Nó cũng không phải tracking ID: tracking ID theo dõi một vật qua nhiều khung hình video, còn instance_id chỉ có nghĩa trong một ảnh tĩnh.
4.  **quy tắc vẽ biên mask** : Mỗi vật một polygon khép kín, đi sát theo đường biên của phần nhìn thấy của vật đó, không thừa nền, không cắt mất phần vật và không lấn sang vật khác chạm bên cạnh.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che: Các bát và khay bột ở góc dưới bên trái chồng lên nhau, bị bàn che một phần. Dùng guideline vì ca này lặp lại ở nhiều ảnh: mask chỉ phủ phần nhìn thấy, ranh giới vẽ theo đường tiếp xúc, mỗi vật một instance. Chỉ escalate khi không rõ là một hay hai vật.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | Một lớp cho cả ảnh theo taxonomy ImageNet-1K (ghi đủ `class_id`, `class_name`, `taxonomy_name`) | Ảnh `traffic` nhiều xe buýt nhưng top-1 là `cab` 0.51; taxonomy không có lớp "traffic" (prediction ≠ GT) | Đọc guideline, chọn đúng 1 lớp theo chủ thể rõ nhất; không có lớp phù hợp thì escalate | Đúng taxonomy, đúng cấp ảnh, không chép score thành nhãn |
| Phát hiện vật thể | Mỗi vật: class + box `bbox_xyxy` | Ảnh `kitchen`: ngưỡng 0.60 có 6 box, hạ xuống 0.20 thành 17 box, thêm các box điểm thấp (`spoon`, `potted plant`, `bottle`, `dining table`); `person` 0.61 chỉ thấy một phần người ở mép trái | Vẽ box sát phần nhìn thấy, đủ vật theo guideline; ca mơ hồ thì escalate | Box có sát không, thiếu/thừa/trùng vật, ca bị che khuất |
| Instance segmentation | Mỗi instance: class + polygon + `instance_id` | `kitchen-010` (`bowl` 0.372): vật nhỏ ~28×16 px, điểm sát ngưỡng 0.35, cùng vật detection chấm 0.70 nên khó quyết định giữ hay bỏ | Vẽ polygon bám biên từng instance, mỗi vật một instance | Mask có lấn sang vật khác không, đủ instance cùng lớp, `instance_id` tách đúng |
## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? Chỉ dùng ba ảnh COCO công khai mà notebook đã cố định và kiểm tra checksum; không dùng, tải lên hay chia sẻ dữ liệu nào khác
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, tôi sẽ dừng lại, không tự xử lý tiếp, và báo cho Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [ ] `day1_lab_outputs/classification_predictions.json`
- [ ] `day1_lab_outputs/detection_predictions.json`
- [ ] `day1_lab_outputs/segmentation_predictions.json`
- [ ] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [ ] `day1_lab_outputs/visuals/classification_top5.png`
- [ ] `day1_lab_outputs/visuals/detection_predictions.png`
- [ ] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [ ] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [ ] Đã nộp **link repository** trên LMS
