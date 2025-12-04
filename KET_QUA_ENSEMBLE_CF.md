# Kết Quả Ensemble Collaborative Filtering

## Tổng Quan

Hệ thống đã thực hiện đánh giá **Ensemble Collaborative Filtering** kết hợp 4 phương pháp tốt nhất trên dataset **MovieLens 100k**.

### Dataset
- **Training samples**: 90,570
- **Test samples**: 9,430
- **Dataset**: MovieLens 100k

---

## 4 Phương Pháp CF Được Sử Dụng

1. **Hybrid CF**: Kết hợp User-User CF và Item-Item CF (alpha = 0.5)
2. **CF Z-score**: Sử dụng Z-score normalization thay vì mean-centering
3. **CF Confidence**: Trọng số dựa trên số lượng ratings của neighbors
4. **CF Adjusted Cosine**: Adjusted cosine similarity với trọng số theo số items chung

---

## Kết Quả Các Cấu Hình Ensemble

### 1. Trọng Số Đều
- **Weights**: [0.25, 0.25, 0.25, 0.25]
- **RMSE**: 0.9802

### 2. Ưu Tiên Hybrid CF
- **Weights**: [0.5, 0.2, 0.2, 0.1]
- **RMSE**: 0.9733

### 3. Ưu Tiên Top 3
- **Weights**: [0.4, 0.3, 0.3, 0.0]
- **RMSE**: 0.9756

### 4. Tối Ưu Theo Kết Quả
- **Weights**: [0.35, 0.3, 0.25, 0.1]
- **RMSE**: 0.9769

---

## So Sánh Với Các Model Riêng Lẻ

| Model | RMSE |
|-------|------|
| Hybrid CF (riêng) | 0.9679 |
| CF Z-score (riêng) | 0.9935 |
| CF Confidence (riêng) | 0.9938 |
| CF Adjusted Cosine (riêng) | 0.9943 |

---

## Tổng Kết Kết Quả

### Xếp Hạng Theo RMSE (Thấp Hơn = Tốt Hơn)

| Hạng | Phương Pháp | RMSE | Cải Thiện vs Baseline |
|------|-------------|------|----------------------|
| 🥇 1 | Hybrid CF (riêng) | 0.9679 | +2.66% |
| 🥈 2 | Ưu tiên Hybrid CF | 0.9733 | +2.11% |
| 🥉 3 | Ưu tiên top 3 | 0.9756 | +1.88% |
| 4 | Tối ưu theo kết quả | 0.9769 | +1.75% |
| 5 | Trọng số đều | 0.9802 | +1.42% |
| 6 | CF Z-score (riêng) | 0.9935 | +0.08% |
| 7 | CF Confidence (riêng) | 0.9938 | +0.05% |
| 8 | CF Adjusted Cosine (riêng) | 0.9943 | +0.00% (Baseline) |

**Lưu ý**: Baseline là CF Adjusted Cosine với RMSE = 0.9943

---

## Demo: Recommendation Cho User

Sử dụng cấu hình tốt nhất: **Hybrid CF (riêng)**

### User 0
Top 5 recommended items:
- Item 1367: Predicted rating = 4.77
- Item 1466: Predicted rating = 4.71
- Item 813: Predicted rating = 4.71
- Item 1598: Predicted rating = 4.71
- Item 1499: Predicted rating = 4.65

### User 10
Top 5 recommended items:
- Item 1658: Predicted rating = 5.50
- Item 1471: Predicted rating = 5.20
- Item 1659: Predicted rating = 4.77
- Item 1466: Predicted rating = 4.52
- Item 813: Predicted rating = 4.52

### User 50
Top 5 recommended items:
- Item 1232: Predicted rating = 5.15
- Item 1553: Predicted rating = 4.84
- Item 1271: Predicted rating = 4.67
- Item 1204: Predicted rating = 4.53
- Item 1275: Predicted rating = 4.47

### User 100
Top 5 recommended items:
- Item 1658: Predicted rating = 5.00
- Item 1670: Predicted rating = 4.38
- Item 1659: Predicted rating = 4.28
- Item 1471: Predicted rating = 4.25
- Item 1293: Predicted rating = 4.15

### User 200
Top 5 recommended items:
- Item 1617: Predicted rating = 4.57
- Item 1620: Predicted rating = 4.57
- Item 1625: Predicted rating = 4.30
- Item 1490: Predicted rating = 4.26
- Item 1303: Predicted rating = 4.18

---

## Kết Luận

### Model Tốt Nhất
- **Phương pháp**: Hybrid CF (riêng)
- **RMSE**: 0.9679
- **Cải thiện**: 2.66% so với baseline

### Nhận Xét

1. **Hybrid CF đơn lẻ cho kết quả tốt nhất**: Kết hợp User-User CF và Item-Item CF với alpha = 0.5 đạt RMSE thấp nhất (0.9679).

2. **Ensemble không cải thiện kết quả**: Các cấu hình ensemble đều có RMSE cao hơn Hybrid CF đơn lẻ, có thể do:
   - Hybrid CF đã đủ tốt và không cần kết hợp thêm
   - Trọng số ensemble chưa được tối ưu hoàn toàn
   - Các model khác (Z-score, Confidence, Adjusted Cosine) có hiệu suất kém hơn nên khi kết hợp làm giảm chất lượng

3. **Baseline (CF Adjusted Cosine) yếu nhất**: RMSE = 0.9943, cho thấy phương pháp này không phù hợp với dataset này.

4. **Demo recommendations hoạt động tốt**: Hệ thống đã tạo được các recommendations hợp lý với predicted ratings từ 4.15 đến 5.50.

### Khuyến Nghị

- **Sử dụng Hybrid CF đơn lẻ** thay vì ensemble trong trường hợp này
- Có thể thử nghiệm với các giá trị alpha khác nhau cho Hybrid CF để tối ưu hơn
- Các phương pháp khác (Z-score, Confidence, Adjusted Cosine) có thể hữu ích cho các dataset khác hoặc khi kết hợp với các kỹ thuật khác

---

*Generated from Ensemble CF evaluation results*

