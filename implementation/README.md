# Recommendation System Implementation

Hệ thống recommendation sử dụng Hybrid Collaborative Filtering + Content-Based Filtering cho Travel Activities.

## 📋 Tổng quan

File `hybrid_cf_cb.py` đọc dữ liệu từ CSV files được export từ backend NestJS, train model và xuất recommendations ra CSV file để backend có thể import vào database.

## 🔄 Workflow

### Bước 1: Export dữ liệu từ Backend NestJS

1. **Export ratings** (format ml-100k):
```bash
POST /recommendations/export/ratings
Body: { "outputPath": "exports/ratings.csv" }
```

Output: `exports/ratings.csv`
Format: `user_id,activity_id,rating,timestamp`

2. **Export activities với category features**:
```bash
POST /recommendations/export/activities
Body: { "outputPath": "exports/activities.csv" }
```

Output: `exports/activities.csv`
Format: `activity_id,category_id,cat0,cat1,...,cat19`

### Bước 2: Copy files vào folder rcm/implementation

Copy 2 files đã export vào folder `rcm/implementation/`:
- `ratings.csv`
- `activities.csv`

### Bước 3: Chạy Python script

```bash
cd rcm/implementation
python hybrid_cf_cb.py
```

**Lần đầu tiên chạy:**
- Đọc `ratings.csv` và `activities.csv`
- Train Hybrid CF+CB model
- Lưu model vào cache (`hybrid_model_cache.pkl`)
- Generate recommendations cho tất cả users
- Xuất kết quả ra `recommendations.csv`

**Các lần chạy sau (mặc định):**
- Đọc `ratings.csv` và `activities.csv` (dữ liệu mới)
- **Tự động load model từ cache** (không train lại)
- Generate recommendations với dữ liệu mới
- Xuất kết quả ra `recommendations.csv`

**Train lại model:**
```bash
python hybrid_cf_cb.py --retrain
```

Output: `recommendations.csv`
Format: `user_id,activity_id,predicted_rating`

### Bước 4: Import recommendations vào Backend

Copy file `recommendations.csv` vào backend (ví dụ: `exports/recommendations.csv`)

```bash
POST /recommendations/import
Body: { "filePath": "exports/recommendations.csv" }
```

Backend sẽ:
- Đọc CSV file
- Nhóm theo user, sắp xếp theo predicted_rating
- Lưu top 10 activities cho mỗi user vào database

### Bước 5: Lấy recommendations cho user

```bash
GET /recommendations?topN=10
Headers: Authorization: Bearer <token>
```

## 📁 File Structure

```
rcm/implementation/
├── hybrid_cf_cb.py         # Main Python script
├── README.md               # Hướng dẫn này
├── ratings.csv             # Input: Ratings từ backend (cần copy vào)
├── activities.csv          # Input: Activities từ backend (cần copy vào)
├── recommendations.csv     # Output: Recommendations (sẽ được tạo)
└── hybrid_model_cache.pkl  # Cache: Model đã train (tự động tạo)
```

## 🔧 Requirements

```bash
pip install pandas numpy scikit-learn scipy
```

## 📊 Format Files

### ratings.csv
```csv
user_id,activity_id,rating,timestamp
1,10,5,1234567890
1,20,4,1234567891
2,10,3,1234567892
...
```

### activities.csv
```csv
activity_id,category_id,cat0,cat1,cat2,...,cat19
1,5,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0
2,3,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
...
```

### recommendations.csv
```csv
user_id,activity_id,predicted_rating
1,15,4.85
1,25,4.72
1,35,4.68
...
```

## ⚙️ Configuration

Trong file `hybrid_cf_cb.py`, bạn có thể điều chỉnh các tham số:

- `cf_k=30`: Số neighbors cho Collaborative Filtering
- `cf_alpha=0.5`: Trọng số User-User vs Item-Item CF
- `cb_alpha=0.01`: Alpha cho ElasticNet trong Content-Based
- `cb_l1_ratio=0.5`: L1 ratio cho ElasticNet
- `weight_cf=0.5`: Trọng số cho CF (CB = 1 - weight_cf)

## 📝 Notes

- File Python sử dụng 0-based indexing cho user_id và activity_id trong quá trình tính toán
- Khi xuất CSV, sẽ chuyển lại về 1-based indexing để khớp với database
- Top 10 recommendations cho mỗi user sẽ được lưu vào database với rank từ 1-10
- Nếu user đã rate một activity, activity đó sẽ không được recommend

## 💾 Model Caching

**Mặc định**: Script sẽ **tự động dùng model đã train** (load từ cache), không train lại mỗi lần chạy.

- **Lần đầu tiên**: Train model và lưu vào `hybrid_model_cache.pkl`
- **Các lần sau**: Tự động load model từ cache → **Nhanh hơn**, không cần train lại
- **Train lại khi cần**: Sử dụng flag `--retrain`

**Command line options:**
```bash
# Sử dụng model cache (mặc định)
python hybrid_cf_cb.py

# Train lại model từ đầu
python hybrid_cf_cb.py --retrain

# Chỉ định file paths
python hybrid_cf_cb.py --ratings my_ratings.csv --activities my_activities.csv --output my_recommendations.csv
```

**Lưu ý:**
- Model cache được lưu trong file `hybrid_model_cache.pkl`
- Xóa file cache nếu muốn train lại: `rm hybrid_model_cache.pkl`
- Có thể dùng model cũ với dữ liệu mới (không cần train lại)

## 🐛 Troubleshooting

1. **File không tìm thấy**: Đảm bảo `ratings.csv` và `activities.csv` nằm trong cùng folder với `hybrid_cf_cb.py`

2. **Lỗi encoding**: Script đã xử lý encoding cho Windows terminal, nếu vẫn lỗi hãy kiểm tra encoding của CSV files

3. **Memory error**: Nếu dataset quá lớn, có thể cần giảm số neighbors (`cf_k`) hoặc chia nhỏ dataset

