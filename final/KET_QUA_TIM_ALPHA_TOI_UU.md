# Kết Quả Tìm Alpha Tối Ưu - Hybrid CF

## Tổng Quan

Hệ thống đã thực hiện tìm kiếm giá trị **alpha tối ưu** cho phương pháp **Hybrid Collaborative Filtering** (kết hợp User-User CF và Item-Item CF) trên dataset **MovieLens 100k**.

### Dataset
- **Training samples**: 90,570
- **Test samples**: 9,430
- **Dataset**: MovieLens 100k

### Phương Pháp
- **Hybrid CF**: Kết hợp User-User CF và Item-Item CF
- **Alpha**: Trọng số cho User-User CF (0.0 = 100% Item-Item, 1.0 = 100% User-User)
- **K**: 30 neighbors

---

## Kết Quả Test Các Giá Trị Alpha

### Bảng Kết Quả Chi Tiết

| Alpha | User-User CF | Item-Item CF | RMSE |
|-------|--------------|--------------|------|
| 0.0 | 0% | 100% | 0.9854 |
| 0.1 | 10% | 90% | 0.9786 |
| 0.2 | 20% | 80% | 0.9734 |
| 0.3 | 30% | 70% | 0.9698 |
| 0.4 | 40% | 60% | 0.9680 |
| **0.5** | **50%** | **50%** | **0.9679** ⭐ |
| 0.6 | 60% | 40% | 0.9697 |
| 0.7 | 70% | 30% | 0.9733 |
| 0.8 | 80% | 20% | 0.9787 |
| 0.9 | 90% | 10% | 0.9858 |
| 1.0 | 100% | 0% | 0.9946 |

---

## Xếp Hạng Theo RMSE (Thấp Hơn = Tốt Hơn)

| Hạng | Alpha | User-User CF | Item-Item CF | RMSE | Chênh Lệch |
|------|-------|--------------|--------------|------|------------|
| 🥇 1 | **0.5** | 50% | 50% | **0.9679** | +0.00% |
| 🥈 2 | 0.4 | 40% | 60% | 0.9680 | +0.01% |
| 🥉 3 | 0.6 | 60% | 40% | 0.9697 | +0.18% |
| 4 | 0.3 | 30% | 70% | 0.9698 | +0.20% |
| 5 | 0.7 | 70% | 30% | 0.9733 | +0.56% |
| 6 | 0.2 | 20% | 80% | 0.9734 | +0.57% |
| 7 | 0.1 | 10% | 90% | 0.9786 | +1.11% |
| 8 | 0.8 | 80% | 20% | 0.9787 | +1.12% |
| 9 | 0.0 | 0% | 100% | 0.9854 | +1.81% |
| 10 | 0.9 | 90% | 10% | 0.9858 | +1.85% |
| 11 | 1.0 | 100% | 0% | 0.9946 | +2.76% |

---

## Phân Tích Kết Quả

### Alpha Tối Ưu
- **Giá trị**: 0.5
- **RMSE tốt nhất**: 0.9679
- **Trọng số**: 
  - User-User CF: 50%
  - Item-Item CF: 50%

### Nhận Xét

1. **Cân bằng 50-50 cho kết quả tốt nhất**: Alpha = 0.5 (cân bằng giữa User-User và Item-Item CF) đạt RMSE thấp nhất (0.9679).

2. **Khoảng alpha tốt**: Các giá trị alpha từ 0.3 đến 0.6 đều cho RMSE dưới 0.97, cho thấy khoảng này là tối ưu.

3. **Item-Item CF tốt hơn User-User CF**: 
   - Alpha = 0.0 (100% Item-Item): RMSE = 0.9854
   - Alpha = 1.0 (100% User-User): RMSE = 0.9946
   - Item-Item CF đơn lẻ tốt hơn User-User CF đơn lẻ khoảng 0.9%

4. **Kết hợp tốt hơn đơn lẻ**: Hybrid CF với alpha = 0.5 (RMSE = 0.9679) tốt hơn cả Item-Item CF đơn lẻ (0.9854) và User-User CF đơn lẻ (0.9946).

5. **Độ nhạy với alpha**: 
   - Chênh lệch giữa alpha tốt nhất (0.5) và alpha xấu nhất (1.0) là 2.76%
   - Các giá trị alpha gần 0.5 (0.4, 0.6) chỉ chênh lệch rất ít (< 0.2%)

---

## Demo: Recommendation Cho User (Với Alpha Tối Ưu)

Sử dụng **alpha = 0.5** (50% User-User CF, 50% Item-Item CF)

### User 0
Top 5 recommended items:
- Item 1466: Predicted rating = 4.79
- Item 1499: Predicted rating = 4.79
- Item 813: Predicted rating = 4.79
- Item 1598: Predicted rating = 4.79
- Item 1535: Predicted rating = 4.79

### User 10
Top 5 recommended items:
- Item 1466: Predicted rating = 4.74
- Item 813: Predicted rating = 4.74
- Item 1535: Predicted rating = 4.74
- Item 1188: Predicted rating = 4.65
- Item 1471: Predicted rating = 4.55

### User 50
Top 5 recommended items:
- Item 1494: Predicted rating = 4.86
- Item 1188: Predicted rating = 4.65
- Item 1499: Predicted rating = 4.65
- Item 1466: Predicted rating = 4.65
- Item 813: Predicted rating = 4.65

### User 100
Top 5 recommended items:
- Item 1188: Predicted rating = 4.46
- Item 1466: Predicted rating = 4.46
- Item 1499: Predicted rating = 4.46
- Item 813: Predicted rating = 4.46
- Item 1593: Predicted rating = 4.21

### User 200
Top 5 recommended items:
- Item 813: Predicted rating = 4.52
- Item 1598: Predicted rating = 4.52
- Item 1499: Predicted rating = 4.52
- Item 1466: Predicted rating = 4.52
- Item 1535: Predicted rating = 4.52

---

## Kết Luận

### Kết Quả Tối Ưu
- **Alpha tối ưu**: 0.5
- **RMSE tốt nhất**: 0.9679
- **Trọng số User-User CF**: 50%
- **Trọng số Item-Item CF**: 50%

### Khuyến Nghị

1. **Sử dụng alpha = 0.5** cho Hybrid CF trên dataset MovieLens 100k
2. **Khoảng alpha 0.4 - 0.6** đều cho kết quả tốt (chênh lệch < 0.2%)
3. **Kết hợp User-User và Item-Item CF** cho kết quả tốt hơn so với sử dụng đơn lẻ
4. **Item-Item CF** có hiệu suất tốt hơn User-User CF một chút, nhưng kết hợp cả hai vẫn tốt nhất

### So Sánh Với Các Phương Pháp Đơn Lẻ

| Phương Pháp | RMSE | So với Hybrid CF (alpha=0.5) |
|-------------|------|------------------------------|
| Hybrid CF (alpha=0.5) | 0.9679 | Baseline |
| Item-Item CF (alpha=0.0) | 0.9854 | +1.81% |
| User-User CF (alpha=1.0) | 0.9946 | +2.76% |

**Kết luận**: Hybrid CF với alpha = 0.5 cải thiện **1.81%** so với Item-Item CF đơn lẻ và **2.76%** so với User-User CF đơn lẻ.

---

*Generated from Hybrid CF alpha optimization results*

